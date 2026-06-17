//+------------------------------------------------------------------+
//| PropEA_WebRequestLock.mqh — MT5 allows one WebRequest at a time   |
//+------------------------------------------------------------------+
#ifndef PROPEA_WEBREQUEST_LOCK_MQH
#define PROPEA_WEBREQUEST_LOCK_MQH

#define PROPEA_WR_LOCK_GV "PropEA_WebRequest_Lock"
#define PROPEA_WR_OWNER_GV "PropEA_WebRequest_Owner"
#define PROPEA_WR_LOCK_TTL_SEC 120

static bool g_propea_wr_local_busy = false;

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
