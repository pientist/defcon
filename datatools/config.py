FIELD_SIZE = (105.0, 68.0)

LINEUP_HEADER = [
    "stats_perform_match_id",
    "game_date",
    "contestant_name",
    "player_id",
    "object_id",
    "shirt_number",
    "match_name",
    "formation",
    "advanced_position",
    "mins_played",
    "start_time",
    "end_time",
]

DEFENSE = [
    "concede",
    "prevent",
    "induce_out",
    "interception",
    "tackle",
    "shot_block",
    "keeper_save",
]

DEFCON_HEADER = LINEUP_HEADER + ["attack_time", "defend_time", "defcon", "defcon_normal"] + DEFENSE

# Categories for spadl event types
SPADL_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "non_action",
    "dribble",
    "goalkick",
    "ball_recovery",  # new, incoming
    "ball_touch",  # new, not handled
    "dispossessed",  # new, not handled
    "shield_ball_oop",  # new, not handled
    "ground_duel",  # new, duel
    "aerial_duel",  # new, duel
    "shot_block",  # new, incoming
    "keeper_sweeper",  # new, incoming
]

PASS = [
    "pass",
    "cross",
    "throw_in",
    "goalkick",
    "corner_short",
    "corner_crossed",
    "freekick_short",
    "freekick_crossed",
]

INCOMING = [
    "interception",
    "ball_recovery",
    "shot_block",
    "keeper_save",
    "keeper_punch",
    "keeper_claim",
    "keeper_pick_up",
    "keeper_sweeper",
]

SHOT = ["shot", "shot_freekick"]  # shot_penalty is excluded since it cannot be blocked by a defender

TAKE_ON = ["take_on", "dribble"]

DUEL = ["tackle", "ground_duel", "aerial_duel"]

DEFENSIVE_TOUCH = ["interception", "ball_touch", "shot_block", "keeper_save", "keeper_punch"]

SET_PIECE_OOP = ["throw_in", "goalkick", "corner_short", "corner_crossed"]

SET_PIECE = SET_PIECE_OOP + ["freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]

NOT_HANDLED = [
    "non_action",
    "clearance",
    "ball_touch",
    "bad_touch",
    "dispossessed",
    "shield_ball_oop",
    "foul",
]

# Tasks corresponding to GNN target types
NODE_SELECTION = [
    "intent",
    "oppo_agn_intent",
    "receiver",
    "intent_receiver",
    "dest_receiver",
    "failed_pass_receiver",
    "blocked_shot_receiver",
    "failure_receiver",
]

NODE_BINARY = [
    "intent_scoring",
    "intent_conceding",
    "receiver_scoring",
    "receiver_conceding",
]

GRAPH_BINARY = [
    "intent_success",
    "dest_success",
    "scoring",
    "conceding",
    "dest_scoring",
    "dest_conceding",
    "shot_blocking",
]
