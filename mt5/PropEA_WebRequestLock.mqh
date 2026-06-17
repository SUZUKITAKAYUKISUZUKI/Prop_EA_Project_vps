//+------------------------------------------------------------------+
//| PropEA_WebRequestLock.mqh — MT5 allows one WebRequest at a time   |
//| Fleet turn queue serializes multi-chart Bridge POSTs.             |
//+------------------------------------------------------------------+
#ifndef PROPEA_WEBREQUEST_LOCK_MQH
#define PROPEA_WEBREQUEST_LOCK_MQH

#define PROPEA_WR_LOCK_GV "PropEA_WebRequest_Lock"
#define PROPEA_WR_OWNER_GV "PropEA_WebRequest_Owner"
#define PROPEA_WR_LOCK_TTL_SEC 120

#define PROPEA_WR_TURN_SLOT_GV "PropEA_WR_TurnSlot"
#define PROPEA_WR_TURN_SINCE_GV "PropEA_WR_TurnSince"
#define PROPEA_WR_QUEUE_SIZE 7
#define PROPEA_WR_TURN_STALL_SEC 180

static bool g_propea_wr_local_busy = false;

//+------------------------------------------------------------------+
int PropEA_RequestSlotIndex(const string symbol)
{
   string canonical = symbol;
   StringToUpper(canonical);
   StringReplace(canonical, ".", "");
   StringReplace(canonical, "_", "");
   StringReplace(canonical, "-", "");
   StringReplace(canonical, " ", "");

   string order[] = {"EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "AUDNZD", "EURGBP", "NZDUSD"};
   for(int i = 0; i < ArraySize(order); i++)
   {
      if(StringFind(canonical, order[i]) == 0)
         return i;
   }
   return (int)(ChartID() % PROPEA_WR_QUEUE_SIZE);
}

//+------------------------------------------------------------------+
void PropEA_EnsureTurnQueueInitialized()
{
   if(!GlobalVariableCheck(PROPEA_WR_TURN_SLOT_GV))
      GlobalVariableSet(PROPEA_WR_TURN_SLOT_GV, 0.0);
   if(!GlobalVariableCheck(PROPEA_WR_TURN_SINCE_GV))
      GlobalVariableSet(PROPEA_WR_TURN_SINCE_GV, (double)TimeCurrent());
}

//+------------------------------------------------------------------+
void PropEA_TouchTurnHeartbeat()
{
   GlobalVariableSet(PROPEA_WR_TURN_SINCE_GV, (double)TimeCurrent());
}

//+------------------------------------------------------------------+
int PropEA_LockAgeSec()
{
   if(!GlobalVariableCheck(PROPEA_WR_LOCK_GV))
      return PROPEA_WR_LOCK_TTL_SEC + 1;
   datetime locked_at = (datetime)GlobalVariableGet(PROPEA_WR_LOCK_GV);
   if(locked_at <= 0)
      return PROPEA_WR_LOCK_TTL_SEC + 1;
   return (int)(TimeCurrent() - locked_at);
}

//+------------------------------------------------------------------+
bool PropEA_IsLockHeld()
{
   return GlobalVariableCheck(PROPEA_WR_OWNER_GV);
}

//+------------------------------------------------------------------+
long PropEA_LockOwnerChart()
{
   if(!GlobalVariableCheck(PROPEA_WR_OWNER_GV))
      return 0;
   return (long)GlobalVariableGet(PROPEA_WR_OWNER_GV);
}

//+------------------------------------------------------------------+
void PropEA_ClearWebRequestLockGlobals()
{
   GlobalVariableDel(PROPEA_WR_OWNER_GV);
   GlobalVariableDel(PROPEA_WR_LOCK_GV);
}

//+------------------------------------------------------------------+
void PropEA_AdvanceRequestTurn(const string symbol)
{
   PropEA_EnsureTurnQueueInitialized();
   int my_slot = PropEA_RequestSlotIndex(symbol);
   int active = (int)GlobalVariableGet(PROPEA_WR_TURN_SLOT_GV);
   if(active != my_slot)
      return;
   int next = (active + 1) % PROPEA_WR_QUEUE_SIZE;
   GlobalVariableSet(PROPEA_WR_TURN_SLOT_GV, (double)next);
   PropEA_TouchTurnHeartbeat();
}

//+------------------------------------------------------------------+
void PropEA_ForceReleaseStaleWebRequestLock(const int max_age_sec)
{
   if(!PropEA_IsLockHeld())
      return;

   int age_sec = PropEA_LockAgeSec();
   if(age_sec < max_age_sec)
      return;

   long owner = PropEA_LockOwnerChart();
   PropEA_ClearWebRequestLockGlobals();
   Print("PropEA WebRequest lock force-released stale owner=", owner, " age_sec=", age_sec);
}

//+------------------------------------------------------------------+
int PropEA_HttpGraceSec(const int expected_http_ms)
{
   if(expected_http_ms <= 0)
      return 60;
   return expected_http_ms / 1000 + 45;
}

//+------------------------------------------------------------------+
void PropEA_MaybeRecoverStalledTurn(const int expected_http_ms)
{
   PropEA_EnsureTurnQueueInitialized();
   datetime since = (datetime)GlobalVariableGet(PROPEA_WR_TURN_SINCE_GV);
   datetime now = TimeCurrent();
   int stall_sec = PROPEA_WR_TURN_STALL_SEC;
   if(expected_http_ms > 0)
      stall_sec = (int)MathMax(stall_sec, PropEA_HttpGraceSec(expected_http_ms) + 60);
   if(since <= 0 || (now - since) < stall_sec)
      return;

   if(PropEA_IsLockHeld())
   {
      int lock_age = PropEA_LockAgeSec();
      if(lock_age < PropEA_HttpGraceSec(expected_http_ms))
         return;
   }

   int active = (int)GlobalVariableGet(PROPEA_WR_TURN_SLOT_GV);
   PropEA_ForceReleaseStaleWebRequestLock(PropEA_HttpGraceSec(expected_http_ms));
   int next = (active + 1) % PROPEA_WR_QUEUE_SIZE;
   GlobalVariableSet(PROPEA_WR_TURN_SLOT_GV, (double)next);
   PropEA_TouchTurnHeartbeat();
   Print("PropEA WebRequest turn queue recovered stalled slot ", active, " -> ", next);
}

//+------------------------------------------------------------------+
int PropEA_ComputeFleetTurnWaitMs(const int http_timeout_ms)
{
   int per_chart = http_timeout_ms + 5000;
   return PROPEA_WR_QUEUE_SIZE * per_chart + 45000;
}

//+------------------------------------------------------------------+
int PropEA_ComputeLockWaitMs(const int http_timeout_ms)
{
   return (int)MathMin(http_timeout_ms + 60000, 150000);
}

//+------------------------------------------------------------------+
bool PropEA_WaitForRequestTurn(const string symbol, const int max_wait_ms, const int expected_http_ms)
{
   PropEA_EnsureTurnQueueInitialized();
   int my_slot = PropEA_RequestSlotIndex(symbol);
   int waited = 0;
   const int step_ms = 250;
   while(waited < max_wait_ms)
   {
      PropEA_MaybeRecoverStalledTurn(expected_http_ms);
      int active = (int)GlobalVariableGet(PROPEA_WR_TURN_SLOT_GV);
      if(active == my_slot)
      {
         PropEA_TouchTurnHeartbeat();
         return true;
      }
      Sleep(step_ms);
      waited += step_ms;
   }
   return false;
}

//+------------------------------------------------------------------+
bool PropEA_ClaimWebRequestOwnership(const string symbol, const int http_timeout_ms)
{
   if(g_propea_wr_local_busy)
      return false;

   long my_chart = (long)ChartID();
   int lock_wait_ms = PropEA_ComputeLockWaitMs(http_timeout_ms);
   int grace_sec = PropEA_HttpGraceSec(http_timeout_ms);
   int waited = 0;
   const int step_ms = 250;

   while(waited < lock_wait_ms)
   {
      PropEA_MaybeRecoverStalledTurn(http_timeout_ms);

      if(!PropEA_IsLockHeld())
      {
         GlobalVariableSet(PROPEA_WR_OWNER_GV, (double)my_chart);
         GlobalVariableSet(PROPEA_WR_LOCK_GV, (double)TimeCurrent());
         g_propea_wr_local_busy = true;
         PropEA_TouchTurnHeartbeat();
         return true;
      }

      long owner = PropEA_LockOwnerChart();
      if(owner == my_chart)
      {
         g_propea_wr_local_busy = true;
         PropEA_TouchTurnHeartbeat();
         return true;
      }

      if(PropEA_LockAgeSec() >= grace_sec)
         PropEA_ForceReleaseStaleWebRequestLock(grace_sec);

      Sleep(step_ms);
      waited += step_ms;
   }
   return false;
}

//+------------------------------------------------------------------+
bool PropEA_BeginWebRequestSession(const string symbol, const int http_timeout_ms, int &out_slot)
{
   out_slot = PropEA_RequestSlotIndex(symbol);
   int turn_wait_ms = PropEA_ComputeFleetTurnWaitMs(http_timeout_ms);
   if(!PropEA_WaitForRequestTurn(symbol, turn_wait_ms, http_timeout_ms))
      return false;
   if(!PropEA_ClaimWebRequestOwnership(symbol, http_timeout_ms))
      return false;
   return true;
}

//+------------------------------------------------------------------+
void PropEA_EndWebRequestSession(const string symbol)
{
   PropEA_ReleaseWebRequestLock();
   PropEA_AdvanceRequestTurn(symbol);
}

//+------------------------------------------------------------------+
bool PropEA_TryAcquireWebRequestLock()
{
   if(g_propea_wr_local_busy)
      return false;
   if(PropEA_IsLockHeld())
   {
      if(PropEA_LockOwnerChart() != (long)ChartID())
         return false;
      if(PropEA_LockAgeSec() >= PROPEA_WR_LOCK_TTL_SEC)
         PropEA_ForceReleaseStaleWebRequestLock(PROPEA_WR_LOCK_TTL_SEC);
      else
         return false;
   }
   GlobalVariableSet(PROPEA_WR_OWNER_GV, (double)ChartID());
   GlobalVariableSet(PROPEA_WR_LOCK_GV, (double)TimeCurrent());
   g_propea_wr_local_busy = true;
   return true;
}

//+------------------------------------------------------------------+
bool PropEA_WaitAcquireWebRequestLock(const int max_wait_ms)
{
   int waited = 0;
   const int step_ms = 250;
   while(waited < max_wait_ms)
   {
      PropEA_ForceReleaseStaleWebRequestLock(PROPEA_WR_LOCK_TTL_SEC);
      if(PropEA_TryAcquireWebRequestLock())
         return true;
      Sleep(step_ms);
      waited += step_ms;
   }
   return false;
}

//+------------------------------------------------------------------+
void PropEA_ReleaseWebRequestLock()
{
   long my_chart = (long)ChartID();
   if(PropEA_IsLockHeld() && PropEA_LockOwnerChart() == my_chart)
      PropEA_ClearWebRequestLockGlobals();
   g_propea_wr_local_busy = false;
}

#endif
