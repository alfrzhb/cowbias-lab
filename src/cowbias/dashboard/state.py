"""Pure shared selection transitions."""


def choose_valid(state, key, options):
    if state.get(key) not in options:
        state[key] = options[0] if options else None
    return state[key]


def reset_scope(state, scope):
    if state.get("scope") != scope:
        for key in (
            "pair",
            "foreground",
            "outcome",
            "background",
            "flips",
            "example",
            "case_selection",
            "figure",
            "export_file",
        ):
            state.pop(key, None)
        state["scope"] = scope
