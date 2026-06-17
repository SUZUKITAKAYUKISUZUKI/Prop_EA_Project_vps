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
#define PROPEA_WR_TURN_STALL_SEC 150

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
void PropEA_AdvanceRequestTurn(const string symbol)
{
   PropEA_EnsureTurnQueueInitialized();
   int my_slot = PropEA_RequestSlotIndex(symbol);
   int active = (int)GlobalVariableGet(PROPEA_WR_TURN_SLOT_GV);
   if(active != my_slot)
      return;
   int next = (active + 1) % PROPEA_WR_QUEUE_SIZE;
   GlobalVariableSet(PROPEA_WR_TURN_SLOT_GV, (double)next);
   GlobalVariableSet(PROPEA_WR_TURN_SINCE_GV, (double)TimeCurrent());
}

//+------------------------------------------------------------------+
void PropEA_MaybeRecoverStalledTurn(const int expected_http_ms)
{
   PropEA_EnsureTurnQueueInitialized();
   datetime since = (datetime)GlobalVariableGet(PROPEA_WR_TURN_SINCE_GV);
   datetime now = TimeCurrent();
   int stall_sec = PROPEA_WR_TURN_STALL_SEC;
   if(expected_http_ms > 0)
      stall_sec = (int)MathMax(stall_sec, expected_http_ms / 1000 + 45);
   if(since > 0 && (now - since) >= stall_sec)
   {
      int active = (int)GlobalVariableGet(PROPEA_WR_TURN_SLOT_GV);
      int next = (active + 1) % PROPEA_WR_QUEUE_SIZE;
      GlobalVariableSet(PROPEA_WR_TURN_SLOT_GV, (double)next);
      GlobalVariableSet(PROPEA_WR_TURN_SINCE_GV, (double)now);
      Print("PropEA WebRequest turn queue recovered stalled slot ", active, " -> ", next);
   }
}

//+------------------------------------------------------------------+
int PropEA_ComputeFleetTurnWaitMs(const int http_timeout_ms)
{
   int per_chart = http_timeout_ms + 4000;
   return PROPEA_WR_QUEUE_SIZE * per_chart + 30000;
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
         return true;
      Sleep(step_ms);
      waited += step_ms;
   }
   return false;
}

//+------------------------------------------------------------------+
bool PropEA_TryAcquireWebRequestLock()
{
   if(g_propea_wr_local_busy)
      return false;

   long my_chart = (long)ChartID();
   datetime now = TimeCurrent();

   if(GlobalVariableCheck(PROPEA_WR_OWNER_GV))
   {
      long owner = (long)GlobalVariableGet(PROPEA_WR_OWNER_GV);
      datetime locked_at = 0;
      if(GlobalVariableCheck(PROPEA_WR_LOCK_GV))
         locked_at = (datetime)GlobalVariableGet(PROPEA_WR_LOCK_GV);
      if(owner != my_chart && locked_at > 0 && (now - locked_at) < PROPEA_WR_LOCK_TTL_SEC)
         return false;
      if(owner != my_chart && locked_at > 0 && (now - locked_at) >= PROPEA_WR_LOCK_TTL_SEC)
      {
         GlobalVariableDel(PROPEA_WR_OWNER_GV);
         GlobalVariableDel(PROPEA_WR_LOCK_GV);
      }
   }

   GlobalVariableSet(PROPEA_WR_OWNER_GV, (double)my_chart);
   GlobalVariableSet(PROPEA_WR_LOCK_GV, (double)now);
   Sleep(100);
   if(!GlobalVariableCheck(PROPEA_WR_OWNER_GV))
      return false;
   if((long)GlobalVariableGet(PROPEA_WR_OWNER_GV) != my_chart)
      return false;

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
   if(GlobalVariableCheck(PROPEA_WR_OWNER_GV))
   {
      if((long)GlobalVariableGet(PROPEA_WR_OWNER_GV) == my_chart)
      {
         GlobalVariableDel(PROPEA_WR_OWNER_GV);
         GlobalVariableDel(PROPEA_WR_LOCK_GV);
      }
   }
   g_propea_wr_local_busy = false;
}

#endif
