"""Independent main protocol identities, never inferred from a generic free label."""
from dataclasses import dataclass

SOURCE_COMMIT = 'b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28'

@dataclass(frozen=True)
class AgentProfile:
    name: str
    method_version: str
    trajectory_method: str
    selection_count: int
    pending_count: int
    cw_visible: bool
    source_path: str
    source_blob: str
    source_sha256: str
    source_commit: str = SOURCE_COMMIT
    batch_size: int = 32
    delivery: str = 'all_candidates_one_request'
    status: str = 'IMPLEMENTED_OFFLINE_PROTOCOL; ONLINE_SCIENCE_NOT_MEASURED'
    lineage_note: str = ''
    selection_attempts: int | None = 3

_PROFILES = {
    'free32_cwaware': AgentProfile('free32_cwaware', 'llm_scientist_v2 / CW-visible Nov1 rerun', 'free_cw_sol61_xhigh_rerun32', 32, 0, True, 'scripts/studies/run_free_cw_sol61_five_20261101.py', 'deb86570f8d867e273dbd0c2d9ceb5c931e35e58', 'd9f2276833367c735871c0fbe4556f2fbe1b234596737887ea24f5bdd79d2b61', delivery='all_candidates_one_request', lineage_note='Historical Free salt, full pool; actual five-seed CW-aware rerun, not unrun paired COMMON_PROMPT.', selection_attempts=8),
    'free32_nocw': AgentProfile('free32_nocw', 'llm32_nocw_five_seed_historical_control_v1', 'llm32_nocw', 32, 0, False, 'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py', 'f6ef9cfcc0d30dbe4143c5c76459bb52cbe3d0eb', '1a0545d52850f99c3d510edc58749565e56c59f82239842f4f9a38587b605243', delivery='all_candidates_one_request', lineage_note='Initial bounded-three-answer five-seed NoCW override; later unbounded runtime amendments are separate and not claimed; shared positive schema, historical Free salt/memory, no extra unrun prose-ID rule.'),
    'cw16_llm16_screen_v1': AgentProfile('cw16_llm16_screen_v1', 'cw16_llm16 screen', 'cw16_llm16', 16, 16, True, 'src/qgeognn_al/active_learning_v2/llm_screen.py', '5bb55938da4e2259a586e66753d43f0df04d064f', '1c2cb06957c8fcade076e4bb79335519ff14037e51f4fddf683aebc4f7eecf83', delivery='screened_128', lineage_note='128 screened candidates; one recorded answer, deterministic valid-retain/fallback. No individual outcomes.', selection_attempts=1),
    'cw16_llm16_scientist_v2': AgentProfile('cw16_llm16_scientist_v2', 'llm_scientist_v2 / astra_high_full_pool_20261001', 'cw16_llm16_scientist_v2', 16, 16, True, 'src/qgeognn_al/active_learning_v2/scientist_study.py', 'd600d68eeef30900c0ee239697fcd1c25e1e5d36', '5371e429f9d15db5c11c6e6842c08f5d4bbbc8361400004065e50a50eafdbed8', delivery='all_candidates_one_request', lineage_note='Complete remaining pool, DIRECT_PROMPT, per-trajectory scientific feedback; not screen-v1.'),
    'free16_cwgradient_lcmd16': AgentProfile('free16_cwgradient_lcmd16', 'lineage display alias / cw16_llm16_scientist_v2', 'cw16_llm16_scientist_v2', 16, 16, True, 'scripts/studies/plot_five_methods_cw_gradient_20261103.py', '63279a01f22cffcd3ff8a37f2afd3ce6cd19fb73', '04f8a8300a76d0628633350f0952e5f6c823851c3433588a58a07578d635ee4d', delivery='all_candidates_one_request', lineage_note='Plot label free16_lcmd16 explicitly reads cw16_llm16_scientist_v2 frozen path; same frozen protocol, not a new experiment.'),
    'free32_cwaware_paired_v1': AgentProfile('free32_cwaware_paired_v1', 'llm32_nocw_paired_v1', 'llm32_cwaware_control', 32, 0, True, 'src/qgeognn_al/active_learning_v2/nocw_inputs.py', '4da0ca103e176ce183156b6e2f3e730f137d2a66', '9ffee2df0cfa056d7f0903379595bd35b28b11aaad6da26894911b7372d37a8b', delivery='all_candidates_one_request', lineage_note='Original paired-study common Prompt and typed evidence/prose-ID restriction, distinct from later actual five-seed override.'),
    'free32_nocw_paired_v1': AgentProfile('free32_nocw_paired_v1', 'llm32_nocw_paired_v1', 'llm32_nocw', 32, 0, False, 'src/qgeognn_al/active_learning_v2/nocw_inputs.py', '4da0ca103e176ce183156b6e2f3e730f137d2a66', '9ffee2df0cfa056d7f0903379595bd35b28b11aaad6da26894911b7372d37a8b', delivery='all_candidates_one_request', lineage_note='Original paired-study COMMON_PROMPT minus CW description; not substituted for historical five-seed prompts.'),
}

def get_profile(name):
    if isinstance(name, AgentProfile):
        if _PROFILES.get(name.name) != name:
            raise ValueError('unregistered agent profile identity')
        return name
    try:
        return _PROFILES[name]
    except (KeyError, TypeError) as error:
        raise ValueError('unknown agent profile; choose an explicit frozen identity') from error


def profile_names():
    return tuple(_PROFILES)
