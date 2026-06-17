//+------------------------------------------------------------------+
//| PyramidLiveManager.mqh — Live Limit ピラミッド (Python bridge 連携) |
//| L5 BT: bar.close 即時約定 / Live: PYRAMID_LIMIT 指値 + /pyramid/*  |
//+------------------------------------------------------------------+
#ifndef PYRAMID_LIVE_MANAGER_MQH
#define PYRAMID_LIVE_MANAGER_MQH

#include "PropEA_WebRequestLock.mqh"
#include "PropEA_TradeExecution.mqh"

#define PYRAMID_LIVE_MAX_TRACKS 8

struct PyramidLiveTrack
{
   bool     active;
   bool     close_pending;
   string   trade_id;
   string   pyramid_group_id;
   string   setup_type;
   string   symbol;
   string   direction;
   ulong    base_ticket;
   ulong    pending_order_ticket;
   int      entry_bar_index;
   int      current_bar_index;
   int      max_layers;
   datetime last_bar_time;
};

PyramidLiveTrack g_pyramid_tracks[PYRAMID_LIVE_MAX_TRACKS];
string g_pyramid_api_base = "http://127.0.0.1:8000";

void PyramidLive_QueueCloseSession(const int track_idx);
void PyramidLive_CloseSession(const int track_idx, const ulong magic);

//+------------------------------------------------------------------+
void PyramidLive_SetApiBaseFromTradeUrl(const string trade_signal_url)
{
   int pos = StringFind(trade_signal_url, "/trade_signal");
   if(pos >= 0)
      g_pyramid_api_base = StringSubstr(trade_signal_url, 0, pos);
   else
      g_pyramid_api_base = trade_signal_url;
}

//+------------------------------------------------------------------+
string PyramidLive_ApiUrl(const string path)
{
   return g_pyramid_api_base + path;
}

//+------------------------------------------------------------------+
bool PyramidLive_PostJson(const string url, const string body, string &response)
{
   const int timeout_ms = 5000;
   int my_slot = PropEA_RequestSlotIndex(_Symbol);
   if(!PropEA_BeginWebRequestSession(_Symbol, timeout_ms, my_slot))
   {
      Print("PyramidLive_PostJson deferred — fleet/lock session unavailable symbol=", _Symbol);
      return false;
   }

   char post[];
   char result[];
   string result_headers;
   StringToCharArray(body, post, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(post, StringLen(body));

   string headers = "Content-Type: application/json\r\n";
   int status = -1;
   for(int attempt = 0; attempt < 3; attempt++)
   {
      status = WebRequest(
         "POST",
         url,
         headers,
         timeout_ms,
         post,
         result,
         result_headers
      );
      if(status == 200 || status == 409)
         break;
      if(status == 1003 && attempt < 2)
      {
         Sleep(400);
         continue;
      }
      break;
   }
   PropEA_EndWebRequestSession(_Symbol);

   if(status == -1)
   {
      Print("PyramidLive WebRequest failed err=", GetLastError(), " url=", url);
      return false;
   }
   if(status != 200 && status != 409)
   {
      Print("PyramidLive HTTP status=", status, " url=", url,
            " body=", CharArrayToString(result));
      return false;
   }

   response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   return true;
}

//+------------------------------------------------------------------+
int PyramidLive_FindSlot()
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
int PyramidLive_FindByTradeId(const string trade_id)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(g_pyramid_tracks[i].active && g_pyramid_tracks[i].trade_id == trade_id)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
int PyramidLive_FindByGroupId(const string group_id)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(g_pyramid_tracks[i].active && g_pyramid_tracks[i].pyramid_group_id == group_id)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
void PyramidLive_ClearSlot(const int idx)
{
   g_pyramid_tracks[idx].active = false;
   g_pyramid_tracks[idx].close_pending = false;
   g_pyramid_tracks[idx].trade_id = "";
   g_pyramid_tracks[idx].pyramid_group_id = "";
   g_pyramid_tracks[idx].pending_order_ticket = 0;
   g_pyramid_tracks[idx].max_layers = 3;
}

//+------------------------------------------------------------------+
bool PyramidLive_PendingOrderActive(const PyramidLiveTrack &track, const string symbol)
{
   if(track.pending_order_ticket == 0)
      return false;
   if(!OrderSelect(track.pending_order_ticket))
      return false;
   long order_magic = OrderGetInteger(ORDER_MAGIC);
   if((ulong)order_magic != 0 && OrderGetString(ORDER_SYMBOL) != symbol)
      return false;
   ENUM_ORDER_STATE state = (ENUM_ORDER_STATE)OrderGetInteger(ORDER_STATE);
   return (state == ORDER_STATE_PLACED || state == ORDER_STATE_PARTIAL);
}

//+------------------------------------------------------------------+
int PyramidLive_CountSessionPositions(
   const string symbol,
   const ulong magic,
   const PyramidLiveTrack &track
)
{
   int count = 0;
   string prefix = "PropEA_PYR_" + StringSubstr(track.pyramid_group_id, 0, 12);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic)
         continue;

      if(ticket == track.base_ticket)
      {
         count++;
         continue;
      }

      string comment = PositionGetString(POSITION_COMMENT);
      if(StringFind(comment, prefix) == 0)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
bool PyramidLive_AtLayerCap(
   const string symbol,
   const ulong magic,
   const PyramidLiveTrack &track
)
{
   int max_layers = track.max_layers > 0 ? track.max_layers : 3;
   int occupied = PyramidLive_CountSessionPositions(symbol, magic, track);
   if(PyramidLive_PendingOrderActive(track, symbol))
      occupied++;
   return occupied >= max_layers;
}

//+------------------------------------------------------------------+
void PyramidLive_CloseOtherSymbolTracks(
   const string symbol,
   const string keep_trade_id,
   const ulong magic
)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         continue;
      if(g_pyramid_tracks[i].symbol != symbol)
         continue;
      if(g_pyramid_tracks[i].trade_id == keep_trade_id)
         continue;
      Print("PyramidLive closing stale symbol track trade_id=", g_pyramid_tracks[i].trade_id);
      PyramidLive_QueueCloseSession(i);
   }
}

//+------------------------------------------------------------------+
double PyramidLive_ComputeAtr(const string symbol, const ENUM_TIMEFRAMES tf, const int period = 14)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   const int need = period + 2;
   int copied = CopyRates(symbol, tf, 0, need, rates);
   if(copied < need)
      return 0.0;

   double sum_tr = 0.0;
   for(int i = 1; i <= period; i++)
   {
      if(i + 1 >= copied)
         break;
      double high = rates[i].high;
      double low  = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr = MathMax(high - low, MathMax(MathAbs(high - prev_close), MathAbs(low - prev_close)));
      sum_tr += tr;
   }
   return sum_tr / (double)period;
}

//+------------------------------------------------------------------+
double PyramidLive_NormalizePrice(const string symbol, const double price)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
bool PyramidLive_ModifySl(
   const ulong ticket,
   const string symbol,
   const double new_sl,
   const double tp
)
{
   if(!PositionSelectByTicket(ticket))
      return false;

   double cur_sl = PositionGetDouble(POSITION_SL);
   double point  = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.00001;
   if(MathAbs(new_sl - cur_sl) < point * 0.5)
      return true;

   double deal_sl = new_sl;
   double deal_tp = tp;
   long pos_type = PositionGetInteger(POSITION_TYPE);
   if(!PropEA_AdjustSlTpForPosition(symbol, pos_type, deal_sl, deal_tp, true))
   {
      Print("PyramidLive skip ModifySL — invalid stops ticket=", ticket);
      return false;
   }

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol   = symbol;
   request.sl       = PyramidLive_NormalizePrice(symbol, deal_sl);
   request.tp       = PyramidLive_NormalizePrice(symbol, deal_tp);
   request.magic    = PositionGetInteger(POSITION_MAGIC);

   if(!OrderSend(request, result))
   {
      Print("PyramidLive ModifySL failed ticket=", ticket, " retcode=", result.retcode);
      return false;
   }
   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
bool PyramidLive_ModifyGroupSl(
   const string symbol,
   const ulong magic,
   const string pyramid_group_id,
   const double new_sl,
   const double tp
)
{
   bool any = false;
   string prefix = "PropEA_PYR_" + StringSubstr(pyramid_group_id, 0, 12);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic)
         continue;

      string comment = PositionGetString(POSITION_COMMENT);
      if(StringFind(comment, prefix) != 0 && StringFind(comment, "PropEA") != 0)
         continue;

      if(PyramidLive_ModifySl(ticket, symbol, new_sl, tp))
         any = true;
   }
   return any;
}

//+------------------------------------------------------------------+
bool PyramidLive_PlaceLimit(
   const string symbol,
   const string direction,
   const double limit_price,
   const double lot,
   const double sl,
   const double tp,
   const ulong magic,
   const string comment,
   ulong &order_ticket
)
{
   order_ticket = 0;
   if(lot <= 0.0 || limit_price <= 0.0)
      return false;

   double deal_sl = sl;
   double deal_tp = tp;
   double deal_lot = lot;
   if(!PropEA_PrepareOrderVolumeStops(symbol, direction, limit_price, deal_sl, deal_tp, deal_lot, 0.0, true))
      return false;

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action    = TRADE_ACTION_PENDING;
   request.symbol    = symbol;
   request.volume    = deal_lot;
   request.type      = (direction == "BUY") ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_SELL_LIMIT;
   request.price     = PyramidLive_NormalizePrice(symbol, limit_price);
   request.sl        = deal_sl;
   request.tp        = deal_tp;
   request.deviation = 20;
   request.magic     = magic;
   request.comment   = comment;
   request.type_time = ORDER_TIME_GTC;

   int filling = (int)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      request.type_filling = ORDER_FILLING_FOK;
   else if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      request.type_filling = ORDER_FILLING_IOC;
   else
      request.type_filling = ORDER_FILLING_RETURN;

   if(!OrderSend(request, result))
   {
      Print("PyramidLive PlaceLimit failed retcode=", result.retcode, " ", result.comment);
      return false;
   }
   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("PyramidLive PlaceLimit retcode=", result.retcode, " ", result.comment);
      return false;
   }

   order_ticket = result.order;
   Print("PyramidLive Limit placed ticket=", order_ticket, " price=", limit_price, " lot=", deal_lot);
   return true;
}

//+------------------------------------------------------------------+
bool PyramidLive_CancelPending(const ulong order_ticket, const string symbol)
{
   if(order_ticket == 0)
      return true;

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_REMOVE;
   request.order  = order_ticket;
   request.symbol = symbol;

   if(!OrderSend(request, result))
   {
      Print("PyramidLive Cancel failed ticket=", order_ticket, " retcode=", result.retcode);
      return false;
   }
   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
bool PyramidLive_CloseTicket(const ulong ticket, const string symbol, const ulong magic)
{
   if(!PositionSelectByTicket(ticket))
      return false;
   if(PositionGetString(POSITION_SYMBOL) != symbol)
      return false;
   if(PositionGetInteger(POSITION_MAGIC) != (long)magic)
      return false;

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
   request.deviation    = 20;
   request.magic        = magic;
   request.comment      = "PropEA_PYR_partial";
   request.type_filling = ORDER_FILLING_RETURN;

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

   if(!OrderSend(request, result))
   {
      Print("PyramidLive partial close failed ticket=", ticket, " retcode=", result.retcode);
      return false;
   }
   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
bool PyramidLive_ExtractJsonStringAt(
   const string json,
   const int start_pos,
   const string key,
   string &value
)
{
   string pattern = "\"" + key + "\":\"";
   int pos = StringFind(json, pattern, start_pos);
   if(pos < 0)
      return false;
   pos += StringLen(pattern);
   int end = StringFind(json, "\"", pos);
   if(end < 0)
      return false;
   value = StringSubstr(json, pos, end - pos);
   return true;
}

//+------------------------------------------------------------------+
bool PyramidLive_ExtractJsonDoubleAt(
   const string json,
   const int start_pos,
   const string key,
   double &value
)
{
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern, start_pos);
   if(pos < 0)
      return false;
   pos += StringLen(pattern);
   int end = pos;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']')
         break;
      end++;
   }
   value = StringToDouble(StringSubstr(json, pos, end - pos));
   return true;
}

//+------------------------------------------------------------------+
bool PyramidLive_ExtractJsonIntAt(
   const string json,
   const int start_pos,
   const string key,
   int &value
)
{
   double d = 0.0;
   if(!PyramidLive_ExtractJsonDoubleAt(json, start_pos, key, d))
      return false;
   value = (int)d;
   return true;
}

//+------------------------------------------------------------------+
bool PyramidLive_ExecuteSingleAction(
   const string symbol,
   const ulong magic,
   const string block,
   PyramidLiveTrack &track
)
{
   string action = "";
   if(!PyramidLive_ExtractJsonStringAt(block, 0, "action", action))
      return false;

   double limit_price = 0, lot_size = 0, sl = 0, tp = 0;
   int layer_index = 0, pending_ticket = 0;
   PyramidLive_ExtractJsonDoubleAt(block, 0, "limit_price", limit_price);
   PyramidLive_ExtractJsonDoubleAt(block, 0, "lot_size", lot_size);
   PyramidLive_ExtractJsonDoubleAt(block, 0, "sl", sl);
   PyramidLive_ExtractJsonDoubleAt(block, 0, "tp", tp);
   PyramidLive_ExtractJsonIntAt(block, 0, "layer_index", layer_index);
   PyramidLive_ExtractJsonIntAt(block, 0, "pending_order_ticket", pending_ticket);

   string group_id = track.pyramid_group_id;
   PyramidLive_ExtractJsonStringAt(block, 0, "pyramid_group_id", group_id);

   string direction = track.direction;
   PyramidLive_ExtractJsonStringAt(block, 0, "direction", direction);

   if(action == "PYRAMID_MODIFY_SL_ALL")
   {
      return PyramidLive_ModifyGroupSl(symbol, magic, group_id, sl, tp);
   }

   if(action == "PYRAMID_CANCEL")
   {
      ulong cancel_ticket = (ulong)pending_ticket;
      if(cancel_ticket == 0)
         cancel_ticket = track.pending_order_ticket;
      bool ok = PyramidLive_CancelPending(cancel_ticket, symbol);
      if(ok)
         track.pending_order_ticket = 0;
      return ok;
   }

   if(action == "PYRAMID_LIMIT")
   {
      if(PyramidLive_AtLayerCap(symbol, magic, track))
      {
         Print("PyramidLive skip LIMIT — layer cap reached trade_id=", track.trade_id);
         return false;
      }
      if(PyramidLive_PendingOrderActive(track, symbol))
      {
         Print("PyramidLive skip LIMIT — pending order already active ticket=", track.pending_order_ticket);
         return false;
      }
      if(track.pending_order_ticket > 0)
      {
         PyramidLive_CancelPending(track.pending_order_ticket, symbol);
         track.pending_order_ticket = 0;
      }

      string comment = StringFormat("PropEA_PYR_%s_L%d", StringSubstr(group_id, 0, 12), layer_index);
      ulong order_ticket = 0;
      if(!PyramidLive_PlaceLimit(symbol, direction, limit_price, lot_size, sl, tp, magic, comment, order_ticket))
         return false;
      track.pending_order_ticket = order_ticket;
      return true;
   }

   if(action == "PYRAMID_MARKET_FALLBACK")
   {
      if(PyramidLive_AtLayerCap(symbol, magic, track))
      {
         Print("PyramidLive skip MARKET_FALLBACK — layer cap reached trade_id=", track.trade_id);
         return false;
      }
      if(PyramidLive_PendingOrderActive(track, symbol))
      {
         PyramidLive_CancelPending(track.pending_order_ticket, symbol);
         track.pending_order_ticket = 0;
      }
      string comment = StringFormat("PropEA_PYR_%s_L%d", StringSubstr(group_id, 0, 12), layer_index);
      double market_price = (direction == "BUY")
         ? SymbolInfoDouble(symbol, SYMBOL_ASK)
         : SymbolInfoDouble(symbol, SYMBOL_BID);
      double deal_sl = sl;
      double deal_tp = tp;
      double deal_lot = lot_size;
      if(!PropEA_PrepareOrderVolumeStops(symbol, direction, market_price, deal_sl, deal_tp, deal_lot, 0.0, true))
         return false;

      MqlTradeRequest request;
      MqlTradeResult  result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action    = TRADE_ACTION_DEAL;
      request.symbol    = symbol;
      request.volume    = deal_lot;
      request.type      = (direction == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      request.price     = market_price;
      request.sl        = deal_sl;
      request.tp        = deal_tp;
      request.deviation = 20;
      request.magic     = magic;
      request.comment   = comment;

      if(!OrderSend(request, result))
         return false;
      return (result.retcode == TRADE_RETCODE_DONE);
   }

   if(action == "PYRAMID_PARTIAL_CLOSE")
   {
      int pos = 0;
      while(true)
      {
         int tpos = StringFind(block, "\"position_tickets\"", pos);
         if(tpos < 0)
            break;
         int num_pos = StringFind(block, ":", tpos);
         int start = StringFind(block, "[", num_pos);
         int end   = StringFind(block, "]", start);
         if(start < 0 || end < 0)
            break;
         string list = StringSubstr(block, start + 1, end - start - 1);
         string parts[];
         int n = StringSplit(list, ',', parts);
         for(int i = 0; i < n; i++)
         {
            StringTrimLeft(parts[i]);
            StringTrimRight(parts[i]);
            ulong ticket = (ulong)StringToInteger(parts[i]);
            if(ticket > 0)
               PyramidLive_CloseTicket(ticket, symbol, magic);
         }
         break;
      }
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
void PyramidLive_ExecuteActionsFromResponse(
   const string symbol,
   const ulong magic,
   const int track_idx,
   const string response_json
)
{
   if(track_idx < 0 || !g_pyramid_tracks[track_idx].active)
      return;

   int actions_pos = StringFind(response_json, "\"actions\"");
   if(actions_pos < 0)
      return;

   int pos = actions_pos;
   while(true)
   {
      int action_pos = StringFind(response_json, "\"action\":", pos);
      if(action_pos < 0)
         break;

      int block_start = action_pos;
      while(block_start > actions_pos && StringGetCharacter(response_json, block_start) != '{')
         block_start--;

      int block_end = StringFind(response_json, "}", action_pos);
      if(block_end < 0)
         break;

      string block = StringSubstr(response_json, block_start, block_end - block_start + 1);
      PyramidLive_ExecuteSingleAction(symbol, magic, block, g_pyramid_tracks[track_idx]);
      pos = block_end + 1;
   }

   string group_id = g_pyramid_tracks[track_idx].pyramid_group_id;
   PyramidLive_ExtractJsonStringAt(response_json, 0, "pyramid_group_id", group_id);
   if(group_id != "")
      g_pyramid_tracks[track_idx].pyramid_group_id = group_id;
}

//+------------------------------------------------------------------+
bool PyramidLive_RegisterAfterEntry(
   const string symbol,
   const ENUM_TIMEFRAMES tf,
   const string response_json,
   const ulong base_ticket,
   const double entry,
   const double sl,
   const double tp,
   const double lot,
   const ulong magic
)
{
   string trade_id = "";
   string setup_type = "";
   ExtractJsonString(response_json, "trade_id", trade_id);
   ExtractJsonString(response_json, "setup_type", setup_type);
   if(trade_id == "")
   {
      Print("PyramidLive register skip — trade_id missing in Python signal JSON");
      return false;
   }

   int existing = PyramidLive_FindByTradeId(trade_id);
   if(existing >= 0)
   {
      Print("PyramidLive register skip — already active trade_id=", trade_id);
      return true;
   }

   PyramidLive_CloseOtherSymbolTracks(symbol, trade_id, magic);
   double atr = 0.0;
   if(!ExtractJsonDouble(response_json, "exit_atr", atr) || atr <= 0.0)
      atr = PyramidLive_ComputeAtr(symbol, tf, 14);
   if(atr <= 0.0)
      atr = MathAbs(entry - sl);

   string direction = "BUY";
   string action = "";
   if(ExtractJsonString(response_json, "action", action) && action == "SELL")
      direction = "SELL";

   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);

   string body = StringFormat(
      "{\"trade_id\":\"%s\",\"setup_type\":\"%s\",\"symbol\":\"%s\",\"direction\":\"%s\","
      "\"entry\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"atr\":%.6f,\"lot_size\":%.4f,"
      "\"base_ticket\":%I64u,\"entry_bar_index\":0,\"tick_size\":%.8f,\"tick_value\":%.8f}",
      JsonEscape(trade_id),
      JsonEscape(setup_type),
      JsonEscape(CanonicalPair(symbol)),
      direction,
      entry,
      sl,
      tp,
      atr,
      lot,
      base_ticket,
      tick_size,
      tick_value
   );

   string response = "";
   if(!PyramidLive_PostJson(PyramidLive_ApiUrl("/pyramid/register"), body, response))
   {
      Print("PyramidLive register failed trade_id=", trade_id, " setup=", setup_type);
      return false;
   }

   string group_id = "";
   ExtractJsonString(response, "pyramid_group_id", group_id);
   double max_layers_d = 3.0;
   ExtractJsonDouble(response, "max_pyramid_layers", max_layers_d);
   int max_layers = (int)max_layers_d;
   if(max_layers < 2)
      max_layers = 3;

   int slot = PyramidLive_FindSlot();
   if(slot < 0)
   {
      Print("PyramidLive no free track slots");
      return false;
   }

   g_pyramid_tracks[slot].active = true;
   g_pyramid_tracks[slot].close_pending = false;
   g_pyramid_tracks[slot].trade_id = trade_id;
   g_pyramid_tracks[slot].pyramid_group_id = group_id;
   g_pyramid_tracks[slot].setup_type = setup_type;
   g_pyramid_tracks[slot].symbol = symbol;
   g_pyramid_tracks[slot].direction = direction;
   g_pyramid_tracks[slot].base_ticket = base_ticket;
   g_pyramid_tracks[slot].pending_order_ticket = 0;
   g_pyramid_tracks[slot].entry_bar_index = 0;
   g_pyramid_tracks[slot].current_bar_index = 0;
   g_pyramid_tracks[slot].max_layers = max_layers;
   g_pyramid_tracks[slot].last_bar_time = 0;

   Print("PyramidLive registered trade_id=", trade_id, " group=", group_id, " max_layers=", max_layers);
   return true;
}

//+------------------------------------------------------------------+
void PyramidLive_OnNewBar(
   const string symbol,
   const ENUM_TIMEFRAMES tf,
   const ulong magic,
   const double daily_dd_remaining_pct
)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, tf, 0, 1, rates) != 1)
      return;

   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         continue;
      if(g_pyramid_tracks[i].symbol != symbol)
         continue;
      if(g_pyramid_tracks[i].last_bar_time == rates[0].time)
         continue;

      g_pyramid_tracks[i].last_bar_time = rates[0].time;
      g_pyramid_tracks[i].current_bar_index++;

      string bar_time = FormatBarTime(rates[0].time);
      string body = StringFormat(
         "{\"trade_id\":\"%s\",\"bar_index\":%d,\"daily_dd_remaining_pct\":%.4f,"
         "\"bar\":{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%.0f}}",
         JsonEscape(g_pyramid_tracks[i].trade_id),
         g_pyramid_tracks[i].current_bar_index,
         daily_dd_remaining_pct,
         bar_time,
         rates[0].open,
         rates[0].high,
         rates[0].low,
         rates[0].close,
         (double)rates[0].tick_volume
      );

      string response = "";
      if(!PyramidLive_PostJson(PyramidLive_ApiUrl("/pyramid/tick"), body, response))
         continue;

      PyramidLive_ExecuteActionsFromResponse(symbol, magic, i, response);
   }
}

//+------------------------------------------------------------------+
void PyramidLive_NotifyFill(
   const int track_idx,
   const double fill_price,
   const ulong position_ticket,
   const ulong order_ticket,
   const ulong magic
)
{
   if(track_idx < 0 || !g_pyramid_tracks[track_idx].active)
      return;

   string body = StringFormat(
      "{\"trade_id\":\"%s\",\"fill_price\":%.5f,\"position_ticket\":%I64u,\"order_ticket\":%I64u}",
      JsonEscape(g_pyramid_tracks[track_idx].trade_id),
      fill_price,
      position_ticket,
      order_ticket
   );

   string response = "";
   if(!PyramidLive_PostJson(PyramidLive_ApiUrl("/pyramid/fill"), body, response))
      return;

   g_pyramid_tracks[track_idx].pending_order_ticket = 0;
   PyramidLive_ExecuteActionsFromResponse(
      g_pyramid_tracks[track_idx].symbol,
      magic,
      track_idx,
      response
   );
}

//+------------------------------------------------------------------+
void PyramidLive_QueueCloseSession(const int track_idx)
{
   if(track_idx < 0 || !g_pyramid_tracks[track_idx].active)
      return;
   g_pyramid_tracks[track_idx].close_pending = true;
}

//+------------------------------------------------------------------+
void PyramidLive_ProcessPendingCloses(const ulong magic)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active || !g_pyramid_tracks[i].close_pending)
         continue;
      PyramidLive_CloseSession(i, magic);
   }
}

//+------------------------------------------------------------------+
void PyramidLive_QueueCloseForSymbol(const string symbol, const ulong magic)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         continue;
      if(g_pyramid_tracks[i].symbol != symbol)
         continue;
      PyramidLive_QueueCloseSession(i);
   }
}

//+------------------------------------------------------------------+
void PyramidLive_QueueCloseForPosition(const ulong position_id, const string symbol, const ulong magic)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         continue;
      if(g_pyramid_tracks[i].symbol != symbol)
         continue;
      if(g_pyramid_tracks[i].base_ticket > 0 && g_pyramid_tracks[i].base_ticket != position_id)
         continue;
      PyramidLive_QueueCloseSession(i);
   }
}

//+------------------------------------------------------------------+
void PyramidLive_CloseSession(const int track_idx, const ulong magic)
{
   if(track_idx < 0 || !g_pyramid_tracks[track_idx].active)
      return;

   string body = StringFormat(
      "{\"trade_id\":\"%s\"}",
      JsonEscape(g_pyramid_tracks[track_idx].trade_id)
   );
   string response = "";
   PyramidLive_PostJson(PyramidLive_ApiUrl("/pyramid/close"), body, response);
   PyramidLive_ExecuteActionsFromResponse(g_pyramid_tracks[track_idx].symbol, magic, track_idx, response);
   PyramidLive_ClearSlot(track_idx);
}

//+------------------------------------------------------------------+
void PyramidLive_OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest &request,
   const MqlTradeResult &result,
   const ulong magic
)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   ulong deal = trans.deal;
   if(deal == 0)
      return;
   if(!HistoryDealSelect(deal))
      return;

   long entry = HistoryDealGetInteger(deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_IN)
      return;

   long deal_type = HistoryDealGetInteger(deal, DEAL_TYPE);
   if(deal_type != DEAL_TYPE_BUY && deal_type != DEAL_TYPE_SELL)
      return;

   string comment = HistoryDealGetString(deal, DEAL_COMMENT);
   if(StringFind(comment, "PropEA_PYR_") != 0)
      return;

   ulong position_ticket = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
   double fill_price = HistoryDealGetDouble(deal, DEAL_PRICE);
   ulong order_ticket = (ulong)HistoryDealGetInteger(deal, DEAL_ORDER);

   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         continue;
      if(g_pyramid_tracks[i].pending_order_ticket == 0)
         continue;
      if(g_pyramid_tracks[i].pending_order_ticket != order_ticket)
         continue;

      PyramidLive_NotifyFill(i, fill_price, position_ticket, order_ticket, magic);
      break;
   }
}

//+------------------------------------------------------------------+
void PyramidLive_PruneClosedTracks(const string symbol, const ulong magic)
{
   for(int i = 0; i < PYRAMID_LIVE_MAX_TRACKS; i++)
   {
      if(!g_pyramid_tracks[i].active)
         continue;
      if(g_pyramid_tracks[i].symbol != symbol)
         continue;

      bool has_base = false;
      if(g_pyramid_tracks[i].base_ticket > 0 && PositionSelectByTicket(g_pyramid_tracks[i].base_ticket))
         has_base = true;

      if(!has_base)
      {
         for(int p = PositionsTotal() - 1; p >= 0 && !has_base; p--)
         {
            ulong ticket = PositionGetTicket(p);
            if(ticket == 0)
               continue;
            if(!PositionSelectByTicket(ticket))
               continue;
            if(PositionGetString(POSITION_SYMBOL) != symbol)
               continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)magic)
               continue;
            string comment = PositionGetString(POSITION_COMMENT);
            if(StringFind(comment, "PropEA_PYR_" + StringSubstr(g_pyramid_tracks[i].pyramid_group_id, 0, 12)) == 0)
               has_base = true;
         }
      }

      if(!has_base)
         PyramidLive_QueueCloseSession(i);
   }
}

#endif
//+------------------------------------------------------------------+
