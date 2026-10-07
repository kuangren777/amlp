"""Shared definitions for AMLP pilots. Side-effecting tools declared per tool from AgentDojo v1.2 specs
(not by name prefix, which misclassified cancel/reschedule/rename and get_webpage)."""
SIDE = {
    "banking": {"send_money", "schedule_transaction", "update_scheduled_transaction", "update_password", "update_user_info"},
    "slack": {"add_user_to_channel", "send_direct_message", "send_channel_message", "invite_user_to_slack",
              "remove_user_from_slack", "post_webpage", "get_webpage"},   # get_webpage = exfil channel via URL
    "travel": {"create_calendar_event", "cancel_calendar_event", "reserve_hotel", "reserve_car_rental",
               "reserve_restaurant", "send_email"},
    "workspace": {"send_email", "delete_email", "create_calendar_event", "cancel_calendar_event",
                  "reschedule_calendar_event", "add_calendar_event_participants", "append_to_file",
                  "create_file", "delete_file", "share_file"},
}
import os
SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "pilot", "snap", "tw_main_20261005.jsonl")

def side_calls(suite, trace):
    return {t["f"] for t in trace if t["f"] in SIDE[suite]}
