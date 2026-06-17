//+------------------------------------------------------------------+
//| PropEA_TradeExecution.mqh — shared lot caps, stop adjust, pair id |
//+------------------------------------------------------------------+
#ifndef PROPEA_TRADE_EXECUTION_MQH
#define PROPEA_TRADE_EXECUTION_MQH

double g_prop_max_single_position_loss_pct = 3.0;
double g_prop_reference_equity             = 0.0;
double g_prop_max_lot_xauusd               = 0.50;
double g_prop_max_lot_fx                   = 2.00;

//+------------------------------------------------------------------+
void PropEA_ConfigureTradeExecution(
   const double max_single_position_loss_pct,
   const double reference_equity,
   const double max_lot_xauusd,
   const double max_lot_fx
)
{
   g_prop_max_single_position_loss_pct = max_single_position_loss_pct;
   g_prop_reference_equity             = reference_equity;
   g_prop_max_lot_xauusd               = max_lot_xauusd;
   g_prop_max_lot_fx                   = max_lot_fx;
}

//+------------------------------------------------------------------+
string CanonicalPair(const string symbol)
{
   string upper = symbol;
   StringToUpper(upper);
   StringReplace(upper, ".", "");
   StringReplace(upper, "_", "");
   StringReplace(upper, "-", "");
   StringReplace(upper, " ", "");

   string pairs[] = {
      "EURGBP", "GBPUSD", "USDCAD", "AUDNZD", "EURUSD",
      "AUDUSD", "NZDUSD", "XAUUSD", "USDJPY", "AUDJPY"
   };
   for(int i = 0; i < ArraySize(pairs); i++)
   {
      if(StringFind(upper, pairs[i]) == 0)
         return pairs[i];
   }
   return upper;
}

//+------------------------------------------------------------------+
int PropEA_PairRequestStaggerMs(const string symbol)
{
   string canonical = CanonicalPair(symbol);
   string order[] = {"EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "AUDNZD", "EURGBP", "NZDUSD"};
   int base = 0;
   for(int i = 0; i < ArraySize(order); i++)
   {
      if(canonical == order[i])
      {
         base = i * 1200;
         break;
      }
   }
   int chart_jitter = (int)(ChartID() % 7) * 200;
   return base + chart_jitter;
}

//+------------------------------------------------------------------+
double NormalizeLot(const string symbol, const double lot)
{
   double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      step = 0.01;
   double out = MathMax(min_lot, MathMin(max_lot, lot));
   out = MathFloor(out / step) * step;
   return out;
}

//+------------------------------------------------------------------+
double NormalizeTradePrice(const string symbol, const double price)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
int EffectiveMinStopPoints(const string symbol)
{
   int stops_level = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   string canonical = CanonicalPair(symbol);
   int floor_pts = 20;
   if(canonical == "XAUUSD")
      floor_pts = 50;
   else if(canonical == "AUDNZD" || canonical == "EURGBP" || canonical == "NZDUSD")
      floor_pts = 25;
   return MathMax(stops_level + 2, floor_pts);
}

//+------------------------------------------------------------------+
bool StopsValidForDeal(
   const string symbol,
   const string action,
   const double market_price,
   const double sl,
   const double tp
)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.00001;
   double min_dist = EffectiveMinStopPoints(symbol) * point;

   if(action == "BUY")
   {
      if(sl > 0.0 && (sl >= market_price || (market_price - sl) < min_dist))
         return false;
      if(tp > 0.0 && (tp <= market_price || (tp - market_price) < min_dist))
         return false;
   }
   else if(action == "SELL")
   {
      if(sl > 0.0 && (sl <= market_price || (sl - market_price) < min_dist))
         return false;
      if(tp > 0.0 && (tp >= market_price || (market_price - tp) < min_dist))
         return false;
   }
   return true;
}

//+------------------------------------------------------------------+
bool AdjustStopsForDeal(
   const string symbol,
   const string action,
   const double market_price,
   double &sl,
   double &tp,
   const bool log_adjustments
)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.00001;

   int min_points = EffectiveMinStopPoints(symbol);
   double min_dist = min_points * point;

   double orig_sl = sl;
   double orig_tp = tp;
   sl = NormalizeTradePrice(symbol, sl);
   tp = NormalizeTradePrice(symbol, tp);

   if(action == "BUY")
   {
      if(sl > 0.0)
      {
         if(sl >= market_price || (market_price - sl) < min_dist)
            sl = NormalizeTradePrice(symbol, market_price - min_dist);
      }
      if(tp > 0.0)
      {
         if(tp <= market_price || (tp - market_price) < min_dist)
            tp = NormalizeTradePrice(symbol, market_price + min_dist);
      }
   }
   else if(action == "SELL")
   {
      if(sl > 0.0)
      {
         if(sl <= market_price || (sl - market_price) < min_dist)
            sl = NormalizeTradePrice(symbol, market_price + min_dist);
      }
      if(tp > 0.0)
      {
         if(tp >= market_price || (market_price - tp) < min_dist)
            tp = NormalizeTradePrice(symbol, market_price - min_dist);
      }
   }

   if(!StopsValidForDeal(symbol, action, market_price, sl, tp))
   {
      Print(
         "PropEA invalid stops after adjust action=", action,
         " market=", DoubleToString(market_price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         " sl=", DoubleToString(sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         " tp=", DoubleToString(tp, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         " min_points=", min_points
      );
      return false;
   }

   if(log_adjustments && (MathAbs(sl - orig_sl) > point * 0.5 || MathAbs(tp - orig_tp) > point * 0.5))
   {
      Print(
         "PropEA adjusted stops (min_points=", min_points,
         ") sl ", DoubleToString(orig_sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         "->", DoubleToString(sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         " tp ", DoubleToString(orig_tp, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         "->", DoubleToString(tp, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         " ref=", DoubleToString(market_price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS))
      );
   }
   return true;
}

//+------------------------------------------------------------------+
double CalcLotFromRiskBudget(
   const string symbol,
   const double risk_budget,
   const double entry,
   const double sl
)
{
   if(risk_budget <= 0.0)
      return 0.0;
   double sl_distance = MathAbs(entry - sl);
   if(sl_distance <= 0.0)
      return 0.0;

   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0.0 || tick_value <= 0.0)
      return 0.0;

   double loss_per_lot = (sl_distance / tick_size) * tick_value;
   if(loss_per_lot <= 0.0)
      return 0.0;
   return NormalizeLot(symbol, risk_budget / loss_per_lot);
}

//+------------------------------------------------------------------+
double ReferenceEquityForRiskCap()
{
   if(g_prop_reference_equity > 0.0)
      return g_prop_reference_equity;
   return AccountInfoDouble(ACCOUNT_EQUITY);
}

//+------------------------------------------------------------------+
double MaxLotBySinglePositionRule(
   const string symbol,
   const double market_price,
   const double sl
)
{
   double equity = ReferenceEquityForRiskCap();
   if(equity <= 0.0 || g_prop_max_single_position_loss_pct <= 0.0)
      return 0.0;

   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0.0 || tick_value <= 0.0)
      return 0.0;

   double sl_distance = MathAbs(market_price - sl);
   if(sl_distance <= 0.0)
      return 0.0;

   double loss_per_lot = (sl_distance / tick_size) * tick_value;
   if(loss_per_lot <= 0.0)
      return 0.0;

   double max_loss_usd = equity * g_prop_max_single_position_loss_pct / 100.0;
   return NormalizeLot(symbol, max_loss_usd / loss_per_lot);
}

//+------------------------------------------------------------------+
double SymbolHardLotCap(const string symbol)
{
   if(CanonicalPair(symbol) == "XAUUSD")
      return g_prop_max_lot_xauusd;
   return g_prop_max_lot_fx;
}

//+------------------------------------------------------------------+
double ResolveExecutionLot(
   const string symbol,
   const double risk_budget,
   const double market_price,
   const double sl,
   const double python_lot
)
{
   double calc_lot = CalcLotFromRiskBudget(symbol, risk_budget, market_price, sl);
   double py_lot = python_lot > 0.0 ? NormalizeLot(symbol, python_lot) : 0.0;
   double lot = 0.0;
   if(py_lot > 0.0 && calc_lot > 0.0)
      lot = MathMin(py_lot, calc_lot);
   else if(py_lot > 0.0)
      lot = py_lot;
   else
      lot = calc_lot;

   if(py_lot > 0.0 && calc_lot > 0.0 && py_lot > calc_lot * 4.0)
   {
      Print(
         "PropEA WARN python lot(", py_lot,
         ") >> broker calc(", calc_lot, ") — using broker calc"
      );
      lot = calc_lot;
   }

   double cap_rule = MaxLotBySinglePositionRule(symbol, market_price, sl);
   double cap_symbol = SymbolHardLotCap(symbol);
   if(cap_rule > 0.0 && lot > cap_rule)
   {
      Print(
         "PropEA FINTOKEI ", g_prop_max_single_position_loss_pct,
         "% cap: lot ", lot, " -> ", cap_rule,
         " ref_equity=", ReferenceEquityForRiskCap()
      );
      lot = cap_rule;
   }
   if(cap_symbol > 0.0 && lot > cap_symbol)
   {
      Print("PropEA symbol hard cap: lot ", lot, " -> ", cap_symbol);
      lot = cap_symbol;
   }
   return NormalizeLot(symbol, lot);
}

//+------------------------------------------------------------------+
bool PropEA_PrepareOrderVolumeStops(
   const string symbol,
   const string action,
   const double reference_price,
   double &sl,
   double &tp,
   double &lot,
   const double risk_budget = 0.0,
   const bool log_adjustments = true
)
{
   if(lot <= 0.0)
   {
      Print("PropEA_PrepareOrderVolumeStops skip — lot<=0 symbol=", symbol);
      return false;
   }
   if(action != "BUY" && action != "SELL")
      return false;
   if(!AdjustStopsForDeal(symbol, action, reference_price, sl, tp, log_adjustments))
      return false;
   lot = ResolveExecutionLot(symbol, risk_budget, reference_price, sl, lot);
   if(lot <= 0.0)
   {
      Print("PropEA_PrepareOrderVolumeStops skip — lot zero after caps symbol=", symbol);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
bool PropEA_AdjustSlTpForPosition(
   const string symbol,
   const long position_type,
   double &sl,
   double &tp,
   const bool log_adjustments = false
)
{
   string action = (position_type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
   double market = (position_type == POSITION_TYPE_BUY)
      ? SymbolInfoDouble(symbol, SYMBOL_BID)
      : SymbolInfoDouble(symbol, SYMBOL_ASK);
   return AdjustStopsForDeal(symbol, action, market, sl, tp, log_adjustments);
}

#endif // PROPEA_TRADE_EXECUTION_MQH
