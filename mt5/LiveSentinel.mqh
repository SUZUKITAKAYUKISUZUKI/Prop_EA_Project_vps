//+------------------------------------------------------------------+
//| LiveSentinel.mqh — リアルタイム生命維持装置 (Prop EA Live Sentinel)   |
//| 1) サーバー日次リセット  2) Floating DD Terminator  3) Rollover/Spread |
//+------------------------------------------------------------------+
#ifndef LIVE_SENTINEL_MQH
#define LIVE_SENTINEL_MQH

input bool   InpSentinelEnabled           = true;
input double InpSentinelFloatingTriggerPct = 3.75;  // 3.5–4.0% (Fintokei 4.5% 手前)
input double InpSentinelFloatingWarnPct     = 3.50;
input int    InpSentinelRolloverStartHour   = 23;
input int    InpSentinelRolloverStartMin    = 55;
input int    InpSentinelRolloverEndHour     = 0;
input int    InpSentinelRolloverEndMin      = 10;

datetime g_ls_server_day        = 0;
double   g_ls_day_start_balance = 0.0;
double   g_ls_day_start_equity  = 0.0;
double   g_ls_day_high_balance  = 0.0;
double   g_ls_floating_dd_pct   = 0.0;
bool     g_ls_entry_locked      = false;
bool     g_ls_terminator_fired  = false;
bool     g_ls_spread_hold       = false;

//+------------------------------------------------------------------+
datetime LiveSentinel_ServerDay(const datetime server_time)
{
   MqlDateTime dt;
   TimeToStruct(server_time, dt);
   return StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
}

//+------------------------------------------------------------------+
int LiveSentinel_MinutesSinceMidnight(const datetime server_time)
{
   MqlDateTime dt;
   TimeToStruct(server_time, dt);
   return dt.hour * 60 + dt.min;
}

//+------------------------------------------------------------------+
bool LiveSentinel_IsRolloverWindow(const datetime server_time)
{
   int now_min   = LiveSentinel_MinutesSinceMidnight(server_time);
   int start_min = InpSentinelRolloverStartHour * 60 + InpSentinelRolloverStartMin;
   int end_min   = InpSentinelRolloverEndHour * 60 + InpSentinelRolloverEndMin;
   if(start_min <= end_min)
      return (now_min >= start_min && now_min <= end_min);
   return (now_min >= start_min || now_min <= end_min);
}

//+------------------------------------------------------------------+
double LiveSentinel_ComputeFloatingDdPct(const double equity)
{
   if(g_ls_day_start_balance <= 0.0)
      return 0.0;
   double floating_loss = MathMax(0.0, g_ls_day_high_balance - equity);
   return floating_loss / g_ls_day_start_balance * 100.0;
}

//+------------------------------------------------------------------+
void LiveSentinel_ResetDaily(const datetime server_time, const double balance, const double equity)
{
   g_ls_server_day        = LiveSentinel_ServerDay(server_time);
   g_ls_day_start_balance = balance;
   g_ls_day_start_equity  = equity;
   g_ls_day_high_balance  = balance;
   g_ls_floating_dd_pct   = 0.0;
   g_ls_entry_locked      = false;
   g_ls_terminator_fired  = false;
   g_ls_spread_hold       = false;
   Print(
      "LIVE_SENTINEL daily reset | server_day=",
      TimeToString(g_ls_server_day, TIME_DATE),
      " balance=", DoubleToString(balance, 2),
      " equity=", DoubleToString(equity, 2),
      " remaining=5.00%"
   );
}

//+------------------------------------------------------------------+
void LiveSentinel_UpdateExtremes(const double balance, const double equity)
{
   if(balance > g_ls_day_high_balance)
      g_ls_day_high_balance = balance;
   g_ls_floating_dd_pct = LiveSentinel_ComputeFloatingDdPct(equity);
}

//+------------------------------------------------------------------+
bool LiveSentinel_CancelAllPending(const ulong magic)
{
   bool any = false;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0)
         continue;
      if(!OrderSelect(ticket))
         continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != magic)
         continue;

      MqlTradeRequest request;
      MqlTradeResult  result;
      ZeroMemory(request);
      ZeroMemory(result);
      request.action = TRADE_ACTION_REMOVE;
      request.order  = ticket;
      if(OrderSend(request, result))
      {
         Print("LIVE_SENTINEL cancel pending ticket=", ticket);
         any = true;
      }
      else
         Print("LIVE_SENTINEL cancel failed ticket=", ticket, " retcode=", result.retcode);
   }
   return any;
}

//+------------------------------------------------------------------+
bool LiveSentinel_PanicCloseAll(const ulong magic)
{
   bool any = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != magic)
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      long pos_type = PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);

      MqlTradeRequest request;
      MqlTradeResult  result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action       = TRADE_ACTION_DEAL;
      request.symbol       = symbol;
      request.volume       = volume;
      request.position     = ticket;
      request.deviation    = 30;
      request.magic        = magic;
      request.comment      = "LIVE_SENTINEL_PANIC";
      request.type_filling = ORDER_FILLING_IOC;

      if(pos_type == POSITION_TYPE_BUY)
      {
         request.type  = ORDER_TYPE_SELL;
         request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
      }
      else
      {
         request.type  = ORDER_TYPE_BUY;
         request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
      }

      if(OrderSend(request, result))
      {
         Print("LIVE_SENTINEL PANIC CLOSE ticket=", ticket, " symbol=", symbol);
         any = true;
      }
      else
         Print("LIVE_SENTINEL PANIC CLOSE failed ticket=", ticket, " retcode=", result.retcode);
   }
   return any;
}

//+------------------------------------------------------------------+
void LiveSentinel_FireTerminator(const double equity, const ulong magic)
{
   g_ls_entry_locked     = true;
   g_ls_terminator_fired = true;
   Print(
      "LIVE_SENTINEL TERMINATOR | floating_dd=",
      DoubleToString(g_ls_floating_dd_pct, 2), "%",
      " >= trigger=", DoubleToString(InpSentinelFloatingTriggerPct, 2), "%",
      " high_balance=", DoubleToString(g_ls_day_high_balance, 2),
      " equity=", DoubleToString(equity, 2),
      " | PANIC CLOSE + entry lock until server 00:00"
   );
   LiveSentinel_PanicCloseAll(magic);
   LiveSentinel_CancelAllPending(magic);
}

//+------------------------------------------------------------------+
bool LiveSentinel_EntryAllowed(
   const datetime server_time,
   const long spread_points,
   const long max_spread_points
)
{
   if(!InpSentinelEnabled)
      return true;
   if(g_ls_entry_locked || g_ls_terminator_fired)
   {
      Print("LIVE_SENTINEL entry lock active until server 00:00 | floating=", DoubleToString(g_ls_floating_dd_pct, 2), "%");
      return false;
   }
   if(LiveSentinel_IsRolloverWindow(server_time))
   {
      Print("LIVE_SENTINEL rollover block | server_time=", TimeToString(server_time, TIME_MINUTES));
      return false;
   }
   if(spread_points > max_spread_points)
   {
      g_ls_spread_hold = true;
      Print("LIVE_SENTINEL spread block | spread=", spread_points, " max=", max_spread_points);
      return false;
   }
   g_ls_spread_hold = false;
   return true;
}

//+------------------------------------------------------------------+
// Returns true when terminator fired on this tick (panic executed).
bool LiveSentinel_OnTick(const ulong magic)
{
   if(!InpSentinelEnabled)
      return false;

   datetime server_time = TimeCurrent();
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   datetime trading_day = LiveSentinel_ServerDay(server_time);

   if(g_ls_server_day == 0 || trading_day != g_ls_server_day)
   {
      LiveSentinel_ResetDaily(server_time, balance, equity);
      return false;
   }

   LiveSentinel_UpdateExtremes(balance, equity);

   if(g_ls_entry_locked || g_ls_terminator_fired)
      return false;

   if(g_ls_floating_dd_pct >= InpSentinelFloatingTriggerPct)
   {
      LiveSentinel_FireTerminator(equity, magic);
      return true;
   }

   if(g_ls_floating_dd_pct >= InpSentinelFloatingWarnPct)
      Print("LIVE_SENTINEL floating warn | dd=", DoubleToString(g_ls_floating_dd_pct, 2), "% equity=", DoubleToString(equity, 2));

   return false;
}

//+------------------------------------------------------------------+
bool LiveSentinel_ShouldHoldLogicClose(const string symbol, const long max_spread_points)
{
   if(!InpSentinelEnabled || !g_ls_spread_hold)
      return false;
   long spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   if(spread > max_spread_points)
   {
      Print("LIVE_SENTINEL spread hold — skip discretionary close | spread=", spread);
      return true;
   }
   return false;
}

#endif
