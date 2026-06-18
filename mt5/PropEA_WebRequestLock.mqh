//+------------------------------------------------------------------+
//| PropEA_WebRequestLock.mqh — MT5 allows one WebRequest at a time   |
//| Simple global mutex only (no fleet turn queue).                   |
//+------------------------------------------------------------------+
#ifndef PROPEA_WEBREQUEST_LOCK_MQH
#define PROPEA_WEBREQUEST_LOCK_MQH

#define PROPEA_WR_LOCK_GV "PropEA_WebRequest_Lock"
#define PROPEA_WR_OWNER_GV "PropEA_WebRequest_Owner"
#define PROPEA_WR_LOCK_TTL_SEC 120

static bool g_propea_wr_local_busy = false;

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
int PropEA_ComputeLockWaitMs(const int http_timeout_ms)
{
   return (int)MathMin(http_timeout_ms + 60000, 150000);
}

//+------------------------------------------------------------------+
bool PropEA_TryAcquireWebRequestLock()
{
   if(g_propea_wr_local_busy)
      return false;

   long my_chart = (long)ChartID();
   if(PropEA_IsLockHeld())
   {
      if(PropEA_LockOwnerChart() != my_chart)
         return false;
      if(PropEA_LockAgeSec() >= PROPEA_WR_LOCK_TTL_SEC)
         PropEA_ForceReleaseStaleWebRequestLock(PROPEA_WR_LOCK_TTL_SEC);
      else
         return false;
   }

   GlobalVariableSet(PROPEA_WR_OWNER_GV, (double)my_chart);
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

//+------------------------------------------------------------------+
void PropEA_ClearLegacyFleetGlobals()
{
   GlobalVariableDel("PropEA_WR_TurnSlot");
   GlobalVariableDel("PropEA_WR_TurnSince");
}

#endif
