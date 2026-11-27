#!/usr/bin/env python3
"""Show A7 continuation progress, GPUs and the shared Mac Key Pool."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import plistlib
import sqlite3
import stat
import subprocess
import time
from xml.parsers.expat import ExpatError


ROOT = Path("/1/column_chromatography_prediction")
RUNS = ROOT / "purellm_a7/runs"
CONTINUATION = "a7-gpt62-astra-xhigh-ssh-v5-prompt-continuation"
FULL = "a7-gpt62-astra-xhigh-ssh-v6-prompt-full"
DIRECT = "a7-gpt62-astra-xhigh-ssh-v8-direct-one"
DIRECT_887 = "a7-gpt62-astra-xhigh-ssh-v9-direct-one"
REPAIR = "a7-gpt62-astra-xhigh-ssh-v10-targeted-repair"
DIRECT_FULL = "a7-gpt62-astra-xhigh-ssh-v11-direct-full"
BACKGROUND = "a7-gpt62-astra-xhigh-background-v12"
SERIAL_IPV4 = "a7-gpt62-astra-xhigh-ipv4-v13"
IPV4_EXTENSION = f"{SERIAL_IPV4}-2357-extension-01"
IPV4_CONTINUATION = f"{SERIAL_IPV4}-2357-continuation-01"
IPV4_R2_EXTENSION = f"{SERIAL_IPV4}-2357-r2-extension-01"
IPV4_R2_CONTINUATION = f"{SERIAL_IPV4}-2357-r2-continuation-01"
IPV4_JSON_EXTENSION = f"{SERIAL_IPV4}-887-json-extension-01"
IPV4_WS_EXTENSION = f"{SERIAL_IPV4}-6101-ws-extension-01"
IPV4_EXTENSION_DISPATCHERS = {
    157: "dispatch_v13_lane_extension",
    887: "dispatch_v13_lane_extension",
    2357: "dispatch_prepared_ipv4_recovery_2357",
    6101: "dispatch_v13_lane_extension",
}
IPV4_CONTINUATION_DISPATCHERS = {
    887: "dispatch_v13_lane_continuation",
    2357: "dispatch_prepared_ipv4_continue_2357",
    6101: "dispatch_v13_lane_continuation",
}
IPV4_EXTENSION_END_ROUNDS = {157: 4, 887: 7, 2357: 2, 6101: 2}
BOUNDED_SSE_END_ROUNDS = {157: 4, 887: 7, 2357: 3, 6101: 2}
RESILIENT_SEGMENTS = ("ipv4_resilient_sse", "ipv4_resilient_failure", "ipv4_relay_retry",
                      "ipv4_relay_retry_02", "ipv4_relay_retry_03", "ipv4_relay_retry_04", "ipv4_relay_retry_05",
                      "ipv4_relay_retry_06", "ipv4_relay_retry_07", "ipv4_relay_retry_08", "ipv4_relay_retry_09",
                      "ipv4_relay_retry_10", "ipv4_relay_retry_11", "ipv4_relay_retry_12", "ipv4_relay_retry_13", "ipv4_relay_retry_14",
                      "ipv4_relay_retry_15", "ipv4_relay_retry_16", "ipv4_relay_retry_17", "ipv4_relay_retry_18",
                      "ipv4_relay_retry_19", "ipv4_relay_retry_20", "ipv4_relay_retry_21", "ipv4_relay_retry_22",
                      "ipv4_relay_retry_23", "ipv4_relay_retry_24", "ipv4_relay_retry_25", "ipv4_relay_retry_26",
                      "ipv4_relay_retry_27", "ipv4_relay_retry_28", "ipv4_relay_retry_29", "ipv4_relay_retry_30",
                      "ipv4_relay_retry_31", "ipv4_relay_retry_32", "ipv4_relay_retry_33", "ipv4_relay_retry_34",
                      "ipv4_relay_retry_35", "ipv4_relay_retry_36", "ipv4_relay_retry_37",
                      "ipv4_6101_r8_targeted_repair", "ipv4_6101_r9_continuation", "ipv4_6101_r11_continuation",
                      "ipv4_cached_r7", "ipv4_cached_r3", "ipv4_cached_r2_6101", "ipv4_cached_r4_6101",
                      "ipv4_6101_r5_continuation",
                      "ipv4_r8_targeted_repair", "ipv4_r8_cached_correction", "ipv4_887_r9_continuation",
                      "ipv4_887_r13_targeted_repair", "ipv4_887_r13_cached_correction", "ipv4_887_r14_continuation",
                      "ipv4_887_r32_targeted_repair", "ipv4_887_r33_continuation",
                      "ipv4_887_r19_cached", "ipv4_887_r20_continuation") + tuple(
    f"ipv4_relay_retry_{wave:02d}" for wave in range(38, 100)
)
FORMAL_APPROVAL_FLAGS = ("authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training")
RELAY_RETRY_SECONDS = 180
REPAIR_887_R13_RUN = f"{SERIAL_IPV4}-887-r13-targeted-repair-01"
REPAIR_887_R13_CONTRACT = "1e19e1e8c13e5b7b0c630fb7c79787f75ff4ffa885c1c075fb3a641cf49151f0"
REPAIR_887_R13_AUTH = "91a621dedc98c5be52d374d3c40584e9b85257685c1cf6a5b9fd11c91dd916ed"
REPAIR_887_R13_METADATA_SHA = "cc0c839543c128305dd96719bbd1f28768d9a317c8ed2901f5af478f48f38654"
REPAIR_887_R13_ORIGINAL_RESPONSE_CUSTODY = {
    "seed_887/pause.request": "9e8a2593f7abc3064585766ec741308373a39b2d03bbcc165723efefcbd76e90",
    "workers/seed_887.json": "cf0d1fd15220f13344c896b35cd1d01870f520dc439170b45cc41fc249d0ed24",
    "seed_887/round_13/turn_00/attempt_00.terminal.json": "79f9398f843d000b623be350c5905fdf1d21af05dbf9e2b6a4b97b8f9486aad7",
    "seed_887/round_13/turn_00/decision.json": "40a656f25cb2e2f65580797187d6a7aa190a71d6a7a006fbc952b0812abcfe8c",
}
REPAIR_887_R13_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r13_targeted_repair.py":
        "9318d0906768aa6d56493cbf8c36350d44ca7c92b73d2681d5c2b86e185de0f9",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_887_r13_targeted_repair.py":
        "d87d15bcc930730c99e43dc11ad9c64ff7067ec1156764830e1e0d80011e89ec",
}
CACHED_887_R13_RUN = f"{SERIAL_IPV4}-887-r13-cached-correction-01"
CACHED_887_R13_CONTRACT = "fea67a7dbab3a3153d224a4cdc810366f015d05d46df4343a55a2712ac6369f0"
CACHED_887_R13_AUTH = "4ef97ab2a321faf84945bd4a4ee5b82a742f32342f68d4e41b121b93d9209be2"
CACHED_887_R13_CHECKPOINT = "571ac7aee1760bb1c73f16269342083acd30e26de6e21c4ad16125caec0eda97"
CACHED_887_R13_ATTEMPT = "59b26706198b4b239f02fadb5267c13e"
CACHED_887_R13_PROJECTED = "cf78fb629ea49be94be539a05fb4ef96d7ae81121940b68879b81e8a3302f3eb"
CACHED_887_R13_CUSTODY_SHA = "500eae760f7b9b28de0e0ee13c522c2bd4f87b41e71edb0db95e271c6ca1d2e5"
CACHED_887_R13_STATE_SHA = "fa036aaf7f664d46c5f26043316a040fb6e8ebba3f65884e26d77676d08ce6cc"
CACHED_887_R13_MEMORY_SHA = "d56373643194525cf45302bdd83aca10a2bb774447dd32a9c16a49cdf4e24d1b"
CACHED_887_R13_FILES = {
    "cached_correction.json": "e421565b5c353fccbaf818293e45130017a479c45252b5ab8356f914d39a2b93",
    "workers/seed_887.json": "3b91e6d0321e336fbe6f1ceccbd91f83233c9e41ef691b122f4e36704c905635",
    "seed_887/round_13/turn_00/attempt_00.terminal.json": "33b35e3ea84802e402d4b7829202548f32a310b4351eca4f38cca3412cf46aa2",
    "seed_887/round_13/batch_freeze.json": "6038bd1e11b157e833800294d75885117b5d7051f37f5d0394ba0732dce791a9",
    "seed_887/round_13/selection.json": "53f84ec5db82ee64f6e3f53b38187f36f64ff6f81b642c62e35cd7bdff927ac8",
    "seed_887/round_14/source.json": "fd24737bdc63d4c8a93a7b7af81a4b573d42907a8ed5198795a9d8eaae3d6a31",
    "seed_887/round_14/transition.json": "b14bb70220accd998cb9f1c5401773124fcc2c2c033146518f22a868be296c42",
    "seed_887/round_14/fit_000/fit_audit.json": "3f760b7834b7cc359dcc76f8f3aad61e6d8794d29f79af7968dde6522815b29e",
    "seed_887/round_14/fit_000/validation_metrics.json": "930d1faf91e375c108910aaf57c248a7eb2447d49e6abe373aad2c8ff483485f",
    "seed_887/round_14/fit_000/best.pt": CACHED_887_R13_CHECKPOINT,
}
CACHED_887_R13_PAID_FILES = {
    "seed_887/pause.request": "71eaff5600feba3a24aedccec9ed41efc4fd0495ca21dc43d3e01f5ae3602d1c",
    "seed_887/round_13/turn_00/attempt_00.terminal.json": "5acfb0d53f27fc2c8262e453dfaf595e6535b761cd8bb8d70c8f404d37fc6a45",
}
CONTINUATION_887_R14_RUN = f"{SERIAL_IPV4}-887-r14-continuation-01"
CONTINUATION_887_R14_CONTRACT = "1cb3a62bf07bda39f88c82b86fdce3c1a729b0fac35ae016662c3f9d9a3d5485"
CONTINUATION_887_R14_AUTH = "cb91385d64b48dedfbf819b69e7e1a9da0fb99679ac642470f34e0907d991226"
CONTINUATION_887_R14_BUDGET = {"first_logical_requests": 22, "maximum_logical_requests": 64, "maximum_physical_attempts": 128}
CONTINUATION_887_R14_PAYLOAD = "42556b0ac60f818003e8ba9ec24d1ceb3124edee82eb6f2734410029a69d69ca"
CONTINUATION_887_R14_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r14_continuation.py":
        "23c1bf5d7ab0ca8d57d761932f291b396a1fbd8dec2c86b465706c49c68e8c2c",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_887_r14_continuation.py":
        "4fb3a449d06805cde801352df71b3e8d866d955d099ffb31ac23875482b271b6",
}
CACHED_887_R19_RUN = f"{SERIAL_IPV4}-887-r19-cached-01"
CONTINUATION_887_R20_RUN = f"{SERIAL_IPV4}-887-r20-continuation-01"
CONTINUATION_887_R20_CONTRACT = "c5105cdaaeb08d1efac389ce50e30f9a4e063fb9346fd84d8b26d1a2d68ee195"
CONTINUATION_887_R20_AUTH = "a546f4ee7398eb4f16dd0c4c0016f5e1b5028ba14963fdf6366f8eb9b96031f5"
CONTINUATION_887_R20_BUDGET = {"first_logical_requests": 16, "maximum_logical_requests": 46, "maximum_physical_attempts": 92}
CONTINUATION_887_R20_PARSER = "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c"
CONTINUATION_887_R20_CACHED_HELPER = "689af1b0e8a5f7e818d3b64079d3d8d95c00f214ba34cf3ecbac184a5128ecac"
CONTINUATION_887_R20_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r20_continuation.py":
        "0c51954ea04dc7bb432f0ae0c41d6e9e1b19dfad2b6d423d534d0d5065ea052c",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_887_r20_continuation.py":
        "c69cb26f54a3d87bdaca8477685966f03dda5fab9b5821895d679261a6cecc02",
}
CACHED_887_R19_HANDOFF = {
    "contract_sha256": "9cf634746ef96b162909e2e27f5e2bcdffcbe270b0435788b15e9dcd358a37d5",
    "authorization_sha256": "28840abdbf1577a8730a860aad0e69496da29c2c47296a0c72b1c61a173b0c20",
    "checkpoint_sha256": "4aff86682563b7f50d7f251605e661ae65897c2f2580d3d6d4ada9130ece5b5e",
    "transition_file_sha256": "bed902db127e3592ae8ba249c587313761413e51194c120399a2033e04d44889",
    "source_file_sha256": "79f91d8382922a05f29a239176e7e9e20b48756ee3a3813e77f152cb640d3ae6",
    "state_sha256": "394fe3b0734e05e73e1392469dc90120636dfbcdf85557a8de76851b78b6d337",
    "scientific_memory_sha256": "f9e9df2076dcab08989b4b8593533ac16824c020bc38b210cb04214d1af31a7e",
    "fit_audit_file_sha256": "6d1c856322ce5425c6ba342193f3c0eda831c54ae21cda5dc523be49372001fe",
    "freeze_sha256": "a2bc6929408d4e1b8aa28bd2c43684067bdcf92beeb92f3a5561979088c2dc9f",
    "cached_selection_sha256": "79e2aec35cfb01139edd803f71c266d0e9a36f0f549c7e0c15ed3021a4863d82",
    "local_terminal_sha256": "a009103826e344c35a1c22aea6bb44ced424c7ca1f36e0cff5b8ae40299e1e3f",
    "local_attempt_id": "c58b67811ed748818110b4ace8109fba",
    "archived_provider_response_id": "resp_ft_b3057b24d2204c009332156fc0aa411c",
    "candidate_order_sha256": "bdc2cbebfce55b823eeeffa453dc0c2d15b185075ea50d8e2243a3aaaee12be8",
    "provider_post": False,
}
CACHED_887_R19_FILES = {
    "seed_887/round_19/batch_freeze.json": "a2bc6929408d4e1b8aa28bd2c43684067bdcf92beeb92f3a5561979088c2dc9f",
    "seed_887/round_19/selection.json": "da97a1fc6170a23ebca4897394be358d0e1fce3b954f4e8e2c79423f22b208b9",
    "seed_887/round_20/source.json": "79f91d8382922a05f29a239176e7e9e20b48756ee3a3813e77f152cb640d3ae6",
    "seed_887/round_20/transition.json": "bed902db127e3592ae8ba249c587313761413e51194c120399a2033e04d44889",
    "seed_887/round_20/fit_000/fit_audit.json": "6d1c856322ce5425c6ba342193f3c0eda831c54ae21cda5dc523be49372001fe",
    "seed_887/round_20/fit_000/validation_metrics.json": "fc123beddcf77b3fcdd3131b4e13a2cba681e3c6e7e482b26abbda79af7a4299",
    "seed_887/round_20/fit_000/fit_device.json": "1bf0cad5304adcacd2d9cbf271288b3abd72815b0f5cb5eac8131b899a57b2ab",
    "seed_887/round_19/turn_00/attempt_00.terminal.json": "a009103826e344c35a1c22aea6bb44ced424c7ca1f36e0cff5b8ae40299e1e3f",
    "cached_response_import.json": "e4cc98e51ed137b83f4b1a939a9a25c6af5843ea2872da9e1515e3e9a54b89de",
    "seed_887/round_20/fit_000/best.pt": "4aff86682563b7f50d7f251605e661ae65897c2f2580d3d6d4ada9130ece5b5e",
}
CONTINUATION_887_R20_REQUEST = {
    "payload_sha256": "aeb44e5e8d85845e7bad1ea14bad6467eb64de837c4829f6ed1d9e75c1671ee4",
    "checkpoint_sha256": "4aff86682563b7f50d7f251605e661ae65897c2f2580d3d6d4ada9130ece5b5e",
    "packet_sha256": "4f15bf9e504d4252806ca9752d706c16653e7e25a77f3afac121affb08013367",
    "wire_packet_sha256": "aa0013a09bfbc18d3342fccdef555a73f7687a092cabbfe5e3f327f6328b0c34",
    "base_messages_sha256": "43d9ed96925fbb215122285e587fd8d2aab2dba1e09ec2fb3d83ea009be6b392",
    "authoritative_base_messages_sha256": "664cb38d346e988a02535f5e70be9f8fdd6b51df53c18bd5b7206e8e9dc613cb",
    "prompt_sha256": "0872b8ddd262b319d9a1463ca0696fa28a5afbd51da8cd3ff477729db6b08d40",
    "mapping_sha256": "08ce40569025d9b259335705f5c6a4dcf16ba8ef1a19a8e372f5649191396e9c",
    "split_sha256": "0413fb5c356dc14b1d2c056379ef09f1925835e64fa41fa86dc12aa8703a1807",
    "state_sha256": "79881eaae89a1e01ea3f3bce861fea63a394289ecd9feed2b6a7a81796c07482",
}
RELAY_RETRY_QUEUE_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/dispatcher.py":
        "cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc",
}
RELAY_157_WAVE01_CONTRACT = "51022ae153326c4d3259ebb183632564184671955e3ca86646debdfc9615862d"
RELAY_157_WAVE01_AUTH = "0ffd0471f4a5ed444f5e67a3c0bbf0e956c7c2fcaaec49a7d64840fff6a27a45"
RELAY_157_NOT_BEFORE = 1795683853.804615
RELAY_157_BOUNDARY_REQUEST = "a39894954355f629d0fb102a9bf04086d6c4235a3d54a4e90350042bbea429a8"
RELAY_157_WAVE02_CONTRACT = "a277fb5b7a0806c2f8f69443ad4b05821659783b8fa65d0b9ba6f7c814f052cd"
RELAY_157_WAVE02_AUTH = "498d339f217f9b4e1d3d4b16cd1e4602048e3ad15715bca927fe1d6c41b797aa"
RELAY_157_WAVE03_PROOF = {
    "payload_sha256": "594ff66ced4fe668cdde6bcbdcc07e9f8cf49933a45accc89ef81a4c97e42107",
    "previous_request_id": "d8fed3a9c76a3a4bc6fa27bac6a705b202b8d90d9565606ed65b1783046feb52",
    "previous_terminal_file_sha256": "0dcc718d9aa7ca2c4b6df268e04e6644c96103d9acd17c90f058ba0631b472cf",
    "previous_transport_progress_file_sha256": "b0b8b6ad3a6366329afb407ed8f6025d26983e594389dd032946ce8976d8dd7c",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795684354.7931683 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 33, "maximum_logical_requests": 97, "maximum_physical_attempts": 193},
}
RELAY_157_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "ddd2a7ca64a74a75a20d198d72501cb1",
    "job_sha256": "dabdd32ea8889eb86fd6ff2228ab66b8b6a3dc149a7087fbc839c2205bcdadd8",
    "result_sha256": "5f0b04be5af4d1476810de3335c077034074e0e101d93f443ee9a2af3a6f2283",
    "error": "sse_frame_error", "provider_dispatched": True, "ambiguous": True,
}
RELAY_157_WAVE03_CONTRACT = "269543cf13a42ca75542f5fd24311fbe437b0e30eb18c8b9c3711ae7b4b0aea6"
RELAY_157_WAVE03_AUTH = "d93de895ea293405d4b6095a6399c6ded0d6f6005d660e31d90937dbbb7b535c"
RELAY_157_WAVE04_PROOF = {
    **RELAY_157_WAVE03_PROOF,
    "previous_request_id": "f6c9b03c951b91dcb38b42364a05dbb24e12af8e45eb34cb365ccdcada2d564e",
    "previous_terminal_file_sha256": "4cb10ccd3f8e9e487554d81b354d6c24f1a263758e3b87f9ef3e1fbfa7753159",
    "previous_transport_progress_file_sha256": "01437d7c316baec695cef4512effb38f539b5f404b333950fb034f41746c4d36",
    "relay_not_before_unix": 1795685449.213155 + RELAY_RETRY_SECONDS,
}
RELAY_157_WAVE03_ATTEMPT = {
    "attempt": 0, "attempt_id": "3a466e59049640aba687e95c19dc5054",
    "job_sha256": "a6e211ffbc55414081fad2f38ac71c9c14f86535d730f7bdc6fc3be80afdb9db",
    "result_sha256": "42b39ab5902e340d0df666232c5ac641185e50d6cf2e62067849cc271bdc9d83",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_157_WAVE04_CONTRACT = "5445e576c203814df1aadad7b761d98405d5acc80894733b8840c45ece3aa2ac"
RELAY_157_WAVE04_AUTH = "56b7acaef8144e4af34fd9cc90c33c815d00f2162be43c60bd23b1d1653b7be2"
RELAY_157_WAVE05_PROOF = {
    **RELAY_157_WAVE04_PROOF,
    "previous_request_id": "e9c640564d39591d242226f44d1bf60a408d92a67dc59c83336b9ae645a43f71",
    "previous_terminal_file_sha256": "af97c1b4949373341e7d7c13121db119c79283515c5de768629f9b9ed300aef2",
    "previous_transport_progress_file_sha256": "d65d849a9fea1c225073c00df258c6439a40818312e35fdc91553a59ef7aa17e",
    "relay_not_before_unix": 1795686187.2452364 + RELAY_RETRY_SECONDS,
}
RELAY_157_WAVE04_ATTEMPT = {
    "attempt": 0, "attempt_id": "8135ea87aba647e2af90541b2455cb43",
    "job_sha256": "39bc1fa4acf97e2366dea34b6aa42b319f27422fcd0d1193b51ebe895e3d9208",
    "result_sha256": "051c4598f9bb78b273188c40d1a3684f1373cf3c423c1da2946da06b9a803fa2",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_157_WAVE05_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "db357841-73d5-4ff8-9e35-976dfaac3a86",
    "response_id": "resp_05dc474a5f199ecb016b07fd86c7d487d29ee2782852ebf58d",
    "conflicting_response_id": "resp_db35784173d54ff89e35976dfaac3a86",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_157_WAVE05_CONTRACT = "360ce022071f58f7b45c80e6554a69bb4433e58cea803bd1f03efafe5e28de50"
RELAY_157_WAVE05_AUTH = "c78dc92578f193670200e5abda1d4f76357b29e419e174916fe16e28df63a584"
RELAY_157_WAVE06_PROOF = {
    **RELAY_157_WAVE05_PROOF,
    "previous_request_id": "1fb54c4cd562bbd605c70778c6e56c6b36e4d30547af99bf82f41e21dfb38137",
    "previous_terminal_file_sha256": "c0e6b8d8d1aae5048b22aa62eb64ac0833981b2c49086ab33f07b312bffcf506",
    "previous_transport_progress_file_sha256": "950630f627fbbbd4a995506248fa32ea9ea1ff34f903fd3764c6a55da4ffc31f",
    "relay_not_before_unix": 1795686851.0807147 + RELAY_RETRY_SECONDS,
}
RELAY_157_WAVE05_ATTEMPT = {
    "attempt": 0, "attempt_id": "f9946e76e81e432997a39dcfdcc6867a",
    "job_sha256": "84713abfe1eca6c3035570210640f432224455e5510c36ddcf9a0c7e7327c3c0",
    "result_sha256": "62caa4e7ba18d9a67b8aa259bb0a5100769df181c88da75e5434b7a11e4cc6c7",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_157_WAVE06_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "1310ea7a-5244-48d2-a158-fd30d4ec4db2",
    "response_id": "resp_02f8dae9ee92f0d3016b0800fb4f2487d2b2ec67c61d50888a",
    "conflicting_response_id": "resp_1310ea7a524448d2a158fd30d4ec4db2",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE01_CONTRACT = "1f1f225de4f396e639ba97bd56b2ab145307e66626b316b115ba7c82beaf11f5"
RELAY_12203_WAVE01_AUTH = "77b8f85d33eed696a3ec19147804a00a963b37bea56b3fcdb2eb5d6dfbf0f16d"
RELAY_12203_NOT_BEFORE = 1795683358.4218476
RELAY_12203_WAVE02_CONTRACT = "f373999a7c88aae8134bf2db212f6c52c8c7d75aedc1779cf5d4daede4b7a86c"
RELAY_12203_WAVE02_AUTH = "fe8cbc69384fb5561d63a91a72173882462d70ddaa87796990e8da24caae7680"
RELAY_12203_WAVE03_NOT_BEFORE = 1795683909.169368 + RELAY_RETRY_SECONDS
RELAY_12203_WAVE03_PROOF = {
    "payload_sha256": "0bd98eeac5b9b9049b7e00d7e8e439d0d1c07e7bd419370440d0259dfc91e1f5",
    "previous_request_id": "3bd1d62bce670c46ad2b10584769a0bfdce2fa9d0d9256000a3b7bbdf1441b79",
    "previous_terminal_file_sha256": "d62cef08ea21bbb26fa238eb1994093c15dc79aeacac9948f5b8f3bbb7130099",
    "previous_transport_progress_file_sha256": "cbda54406f564c8869a77f1ad09de5ada161f9b03d02cabb621bc367009bcfbd",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": RELAY_12203_WAVE03_NOT_BEFORE,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187},
}
RELAY_12203_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "ff520a3b471840e68a1f9ed34bee41f1",
    "job_sha256": "0fafc1c0274a578b04eb5a21ac9a78981d72fefecfa313455d2b814a4f3516ad",
    "result_sha256": "898014f9ac17cf60b55317aae8e4dadb56243c9ea689e1883b2c1c8044327803",
    "error": "connection_interrupted", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE03_CONTRACT = "a62436a562a1007d441cd88aac5f92e2c53609a4232d49d4c6d87410dbe49598"
RELAY_12203_WAVE03_AUTH = "886e8993730af0dcafa5ce286e0e7701a6397e778324db72e48053efdf93b46a"
RELAY_12203_WAVE04_PROOF = {
    **RELAY_12203_WAVE03_PROOF,
    "previous_request_id": "b1dd1488f577af8ede100513f1f25647bbe5c4837d3e377d8cb2b7c0aef18722",
    "previous_terminal_file_sha256": "0dd0f1b70b18755e4c825f705247cdc6f2aed747d78c565ce08cee555f912dc2",
    "previous_transport_progress_file_sha256": "5c8b9bb2953d7232f92e7d4cf9592113706d39f08187e9bb236d7cd72dc7f309",
    "relay_not_before_unix": 1795685708.5045788 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE03_ATTEMPT = {
    "attempt": 0, "attempt_id": "7f4f602283a14b5ebd89fdb98247e306",
    "job_sha256": "ae6b8232e425c1f74afc6125588558c9b5a09b0be7774ee49d662d2df767ae5c",
    "result_sha256": "03b6c645d58e3f42285b5d176bfeec3606a582f915b01a7d67e4ae55a30e556f",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE04_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "sse_event_count": 16, "provider_request_id": "b5b53dcc-c91b-42dd-8be1-4e689bd3f5f3",
    "response_id": "resp_0377120a5eb95a6d016b07f8c54f7887d2bf3179451cf31d5a",
}
RELAY_12203_WAVE04_SOURCES = RELAY_RETRY_QUEUE_SOURCES
RELAY_12203_WAVE04_CONTRACT = "5650df1c3ec2bfc02b3d414263bd99753f9882c93d8bc1910102cf262451055e"
RELAY_12203_WAVE04_AUTH = "e0772b0349f742db1934f33beedfc83c45d2f8b0cd043512257651d0e44bd4f9"
RELAY_12203_WAVE05_PROOF = {
    **RELAY_12203_WAVE04_PROOF,
    "previous_request_id": "078866fe9c060d20a54586a439458673db6c26d43ff88fa626c831c152d8a5a0",
    "previous_terminal_file_sha256": "67de43373ff29535f34bdc399ff499f8033a33c20efc6ef6a399cf4bb234b1fe",
    "previous_transport_progress_file_sha256": "37c6dba9fdda950fb0c3179552026ed6b8a3df61eb31e05ef9368445c1dec665",
    "relay_not_before_unix": 1795686451.363517 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE04_ATTEMPT = {
    "attempt": 0, "attempt_id": "57b87f54c0ce486c83b7d0a5ffcb1699",
    "job_sha256": "d97c642d070c3d7524a45460ff913e210cc852575897ddde55fd5aaf916180ef",
    "result_sha256": "dc3b965716c613d665136853bc6aa4a67a021c66ab6483645f913fef4d8813f1",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE05_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "2df9333e-c775-46a7-8dfc-89aa54cf3749",
    "response_id": "resp_00271412b94c2133016b07ff67b75487d2ba227b0979fa30a1",
    "conflicting_response_id": "resp_2df9333ec77546a78dfc89aa54cf3749",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE05_CONTRACT = "e2b5af09a15ec1c32dcc7ddd3d0f418ac12309178154c304379a0f69dd614a77"
RELAY_12203_WAVE05_AUTH = "435c4957488c881e7132ef1541403328a5b6cf0d0ee383a7e9ff11de5ca400b0"
RELAY_12203_WAVE06_PROOF = {
    **RELAY_12203_WAVE05_PROOF,
    "previous_request_id": "29f5cd07c5babf292526190b1d4ffc831e81b4973996e377aec62108241bcdcc",
    "previous_terminal_file_sha256": "d8f625b0cf3f3fc44a8174c81d2e21be6aaa984c5d2fa69fd0ceef1ef585610a",
    "previous_transport_progress_file_sha256": "c68e7df78ed3a63c4cf2ec4982b00ff1a5eb6649d920e391728f9485a7772824",
    "relay_not_before_unix": 1795687035.0316763 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE05_ATTEMPT = {
    "attempt": 0, "attempt_id": "e729a3ed4b8646eeb4f138afa5cdfd92",
    "job_sha256": "44ea8435296a6b7135a027d4ef7ac5f6b13be2b3cc308a8e6b84caf1c1e667f9",
    "result_sha256": "85ab759d4150348aea0821c1b55cc0a4bd77dab56aed4316d5b40ef5b7f92160",
    "error": "error", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE06_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "event_type": "error", "error_type": "error",
    "provider_error_code": "upstream_stream_read_error", "sse_event_count": 10,
    "provider_request_id": "e8a4abba-c6fc-43b4-863a-340949691acb",
    "response_id": "resp_03e8605b24debb20016b080227447887d2ac1a1e5f5313b441",
}
RELAY_6101_WAVE01_CONTRACT = "978ef32b62d20f330b3fd9f4fa40afc9831c098d01a1a2ee013655ae9f18ab4d"
RELAY_6101_WAVE01_AUTH = "f8dc94f7d0c9c68b9023230675fcb1b476864ada0efc136123f991c88f1736f6"
RELAY_6101_FITTED_PREFIX = {
    "round_02/source.json": "c33ecb89a166f5f2d112f2bb434fab6529d4bf108fe0d4f84b28d3b7e39648b0",
    "round_02/transition.json": "90e14a3d1ab7fd8719c68b5ec86c3d3d580cf5c733a880bc6581f371995cf54c",
    "round_01/memory.json": "989f5e746a5398725bd4bcb37ee199fab8b1f13516e0e58c6d47b66365b07cec",
}
RELAY_6101_WAVE01_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "70aeee7f486449bba0e3344bece900cc",
     "job_sha256": "0f7db8326ea1e093e10a9629c42fec235ef0f30a92583041b0d90722d7d93711",
     "result_sha256": "7ad3f71eba6ad06fc442e87d5cbc65e98f99038af24635e983f021c1c07f89c9",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "2547ab46579b4341bc04f26550be98fc",
     "job_sha256": "c1da929e150505f240cf13a6b590f605a6b6dc438bd23893fed4640ee7f7c70d",
     "result_sha256": "755fedb86b30db88d74fb6c9eb45faeb4dd4c017f8b3ccf5d6c4e2bff5bfccfd",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
)
RELAY_6101_WAVE01_TERMINALS = (
    "64a972a3d630c138d0b891eead861eb1ff71ee9c3d3bcab2289022dab6b2ad1d",
    "42e00584598175ae64e7af0a569f7402a7071d7b9c73a5ab373264a91b82512f",
)
RELAY_6101_WAVE02_PROOF = {
    "logical_index": 0,
    "payload_sha256": "f3f3dac061e9baca67095771305e0ea47827c20d35e7d1018c89059172533b1a",
    "previous_request_id": "2f30189c779807c27f260f361864c4a88e777f0677b6d1cdaa81247023803303",
    "previous_terminal_file_sha256": RELAY_6101_WAVE01_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "7b5b71d36c42c24bf39ea4ecce27d11e7b3c8d1dbe2ee57f963c1d97fd82c1bd",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795684592.3081465 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 34, "maximum_logical_requests": 100, "maximum_physical_attempts": 199},
}
RELAY_6101_WAVE02_CONTRACT = "19d8da15a90d182094a65f8693239061a71603958a34a352d58da0ae54aff595"
RELAY_6101_WAVE02_AUTH = "619b51b8177eb1cd071ba03dbb29a1e6c78f837ed7e250123a57cab032f4e1d7"
RELAY_6101_WAVE03_PROOF = {
    **RELAY_6101_WAVE02_PROOF,
    "previous_request_id": "0b921fac494749af3083c5ada05ea22f4be80a973a5e1db455d9811e7d439204",
    "previous_terminal_file_sha256": "2305a231866b9862b33d73c300c5580f05f8cacbc5fa01867c8a9c3e722e018d",
    "previous_transport_progress_file_sha256": "9f558de4e29358510a6f35254a82c3705c93b974eba37d30424ebbc7ed051948",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795686404.2017465 + RELAY_RETRY_SECONDS,
}
RELAY_6101_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "094b37236e0f42b19cf30adcda11deb8",
    "job_sha256": "592dd5651e0ac75f15968d288cba37d53713a9a456f67d9bbbf441781bf5a5d9",
    "result_sha256": "cd93a0e2bb33b38528e8acdee763926d38ece22af79a89c71391ae0176777578",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE03_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "679f9855-2eda-4e86-9ea5-4fe75ed45719",
    "response_id": "resp_02c95f0c40e3ffc3016b07fd73ff1087d199a88a399a8c5583",
    "conflicting_response_id": "resp_679f98552eda4e869ea54fe75ed45719",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_6101_WAVE03_CONTRACT = "250e3b87e46a3013f37af35bbe4a911c4214c8efa419a44b52d0c52c56fe496c"
RELAY_6101_WAVE03_AUTH = "a43dd05f6020dd6b27f5e91f703082817f7f74699e99717e3685abb7d6fa805a"
RELAY_6101_WAVE03_ATTEMPT = {
    "attempt": 0, "attempt_id": "688a8fd0d3524eabb491b7fcaddddd6a",
    "job_sha256": "c0781c7eff611bc630c3bf66c831c4a07410b9b1ced87cc3fe1b5ae69e722e42",
    "result_sha256": "1f73d28e528dd56011f30eb13a09911700480675fc6c1eb1da725386382ea1c1",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE04_PROOF = {
    **RELAY_6101_WAVE03_PROOF,
    "previous_request_id": "d30c0ed67afa97864f63c0c76c73e62a1854747845583c335d110b42b70735f8",
    "previous_terminal_file_sha256": "dc14eca9467812a4b3c58a8c9e6bdd0d77c8bdb0fba8cf7306457a5169becefb",
    "previous_transport_progress_file_sha256": "ca16f3084e0265ffdf2042996a8d646facc69e33901ebdb6af2014a6be021d12",
    "relay_not_before_unix": 1795690616.5505776 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 34, "maximum_logical_requests": 100, "maximum_physical_attempts": 200},
}
RELAY_6101_WAVE04_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta", "sse_event_count": 3016, "last_sse_seconds": 3531.8199360000435,
    "provider_request_id": "bff33246-96b7-4c53-8bf2-8e477b1f16ba", "model": "gpt-6.2-astra",
    "response_id": "resp_07eba1f236437c1c016b08027af25487d0981f8a128889e114",
}
RELAY_6101_WAVE04_CONTRACT = "d91884196aced55e77bb9e9cd2c7d790653060553c5cf573760b092fcc05d303"
RELAY_6101_WAVE04_AUTH = "54215e67efd44263b9b4956d03dc26b28a7914cf6f318e75ff851febf8c42e53"
RELAY_6101_WAVE05_CONTRACT = "e391fb682d43acd1e452fc3a09e7be5596274cee26e23ee1d1f4f91e0ae69a0d"
RELAY_6101_WAVE05_AUTH = "2f4aae25cfaf9bfcbdf98063cc2f701138274cf46c756ff2825ab8798e705219"
RELAY_6101_WAVE05_PROOF = {
    **RELAY_6101_WAVE03_PROOF,
    "previous_request_id": "45a83e9d69c4b3c4e5abbfb48d2007fc032ab9cc9d1cc711a7c276aa4f62d1a5",
    "previous_terminal_file_sha256": "b1e38d716655fe6e3d6716bf91d48c7f127845f12f9f047693e92236309cf61b",
    "previous_transport_progress_file_sha256": "060a67cb60da1119d693d2b800c914d6a46acd9b7ecb5a5b8be26670d43d7d25",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795692351.0271559 + RELAY_RETRY_SECONDS,
}
RELAY_6101_WAVE04_ATTEMPT = {
    "attempt": 0, "attempt_id": "1b132db24b2c493dacaa226f3b8a9d03",
    "job_sha256": "63673125a094b1307479e6f334b8d4cae3b82d86ac189ee324d6bde74b56c172",
    "result_sha256": "103d35e0e06c00d82f603b71f71deabd462a563d5f74b25e349b69bf4016afae",
    "error": "sse_frame_error", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE05_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "ReadError",
    "event_type": "response.output_text.delta", "sse_event_count": 252, "last_sse_seconds": 1201.01432175003,
    "provider_request_id": "bfe68056-9b30-465c-9a09-a1632d2350c4", "model": "gpt-6.2-astra",
    "response_id": "resp_0fdef8f9826fb9c4016b08126348a087d299ea05eb53d3e6d5",
}
RELAY_6101_WAVE06_CONTRACT = "3380abf0b3fbde69a4d4c7a9d8e64dc815cc64c258736088baa3041fff209023"
RELAY_6101_WAVE06_AUTH = "ae353451f7f46dd0b36a4515e8e469defe36348f50e9be8752999d7999a228c1"
RELAY_6101_WAVE06_PROOF = {
    **RELAY_6101_WAVE04_PROOF,
    "previous_request_id": "9a69c8b4d228bf5a14134f0b4a2eede7b23f7de1e9a33f49109c2d47d185d1fa",
    "previous_terminal_file_sha256": "6ccadea081c37e6fc2695d61c517222d9c42b90874172e3b2005ee56eff211b4",
    "previous_transport_progress_file_sha256": "83f034ab7b6150e487311e0b5a213b5d16a55e5e11d42b4dd8312744f5e4aa73",
    "relay_not_before_unix": 1795692993.4806566 + RELAY_RETRY_SECONDS,
}
RELAY_6101_WAVE05_ATTEMPT = {
    "attempt": 0, "attempt_id": "afcc54a04b93475ba7edcab4471d784e",
    "job_sha256": "cb1b1a191293150babaedfdf77ab9e7bc6c5fb141cc87ada6828679d57659eff",
    "result_sha256": "d9c055dbb26c36730e2072b738665821937a732d02a78292a33fe2c7b97acf37",
    "error": "response.failed", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE06_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response.failed", "event_type": "response.failed",
    "provider_error_code": "server_error", "sse_event_count": 41, "last_sse_seconds": 318.5360944159329,
    "provider_request_id": "42fe0655-daa5-4c5c-863c-7ff3c5a4f33e", "model": "gpt-6.2-astra",
    "response_id": "resp_0361f6be4607ecc5016b0818828c8c87d29fd7a92b30d25ab7",
}
CACHED_R2_6101_CONTRACT = "802ae4d2191ecfb834bfe7bed995f54d9ffe40c36a2ec5cb48236fdc4c97ebad"
CACHED_R2_6101_AUTH = "40098571aeb8b13c3eaa0f0ffb1870ba87f318bbfcdc11218480b5d97dd96c59"
CACHED_R2_6101_CHECKPOINT = "b31e739a950e118df30f2b64b37ba072314ec3164bc83d0bdc2f0518e348f73c"
CACHED_R2_6101_TRANSITION = "8e5f9d4cc67593e008734a4ba72194970560a5d7d495ee4c507299391f2d9996"
CACHED_R2_6101_RESPONSE = "resp_04415e2e4258a85c016b08202301c087d283892240b2871ffd"
CACHED_R2_6101_BUDGET = {"first_logical_requests": 33, "maximum_logical_requests": 99, "maximum_physical_attempts": 198}
CACHED_R2_6101_REVIEW = {
    "attempt_id": "18c94b2993e1444686651504f7aeb992",
    "job_sha256": "f02620a8b8bc34c516dc4a9b2d31b01e065ea2dc1f969c6d2f7c3be12de086cd",
    "result_sha256": "9497827cae957a42049346b25bb15650eb3213bc7db11a95d2e6af9fd83c3e60",
    "raw_response_sha256": "d7f5586ac48d83e1af98faa24ad17317aa200c5148b8037451493f09bea82ea4",
    "normalized_plan_sha256": "c13db811a7e733ebf186dbeb03959c7fe917218c0cd8f244812824669d593789",
    "ordered_candidate_ids_sha256": "7d6f34ae97691850af98ebbc46f14010aec547d24f06f020d916f73f0423ce06",
    "archived_terminal_file_sha256": "b49f95e69d5517ddd433423ee3de039260d28aa40b9aed8cd368a8e69617ea97",
    "archived_decision_file_sha256": "a03701ebfb59012d1e04898b472d414a94bde8be0debf12754b47c8eeb47c778",
    "provider_post": False,
}
CACHED_R2_6101_MIRRORS_SHA = "ccf160994bcb7d187e414728416a56cc05391edebc2264a5e818409b4be6cb69"
RELAY_6101_WAVE07_CONTRACT = "1da70806e319c8cbacff6e16894d1667fd0f7ad859075410c9ef7ccb49f33422"
RELAY_6101_WAVE07_AUTH = "3e4e510b3c06e0fe26fe39e33155ff85d346c48ab5047e0518cca2bbf91ddc15"
RELAY_6101_WAVE07_CACHED_REVIEW = {
    "proof_sha256": "257e1b849cd8466d7eb599784c30ce5fc72f6afc974930b0c88ad3e675a61c57",
    "receipt_file_sha256": "55c1aaa458c9e105ce9d9b94ff7434ccfb347b69ea1b96a335175ecec713b244",
    "resource_contract_sha256": RELAY_6101_WAVE06_CONTRACT, "resource_authorization_sha256": RELAY_6101_WAVE06_AUTH,
    "provider_post": False,
}
RELAY_6101_WAVE07_RESOURCE_SHA = "ad483af2538715ea4f70f6e7bbbdbc01b590058a0b4923ec0595d0719cb3785e"
RELAY_6101_WAVE07_FITTED_FILES = {
    "round_02/turn_00/attempt_00.terminal.json": "cc3ef06bb9e2798bacdfadd15bfd3e5c4a8f95d196463ce5bf08be55c52c82fa",
    "round_02/batch_freeze.json": "ed3d2e410799a947e28302669e15ef2c11f269af6a20d990fa1a3ffc056fa31e",
    "round_02/selection.json": "08a9e6d66e59c3b2497984c4ffab2090d91e72d5cffd6e91beea02298c3490f3",
    "round_02/memory.json": "8d6cd100558f57f85be74755824ef9073b805bfe77a0cfe30ee985fcb5938b92",
    "round_03/source.json": "cea613518df5533dde899b691f5f3db7a1bb2c761dcf18ab30c8f116675b10f9",
    "round_03/transition.json": CACHED_R2_6101_TRANSITION,
    "round_03/fit_000/fit_audit.json": "30b84b35840f7e79999c6a6ff0895ba8fcac504e3c6f059e52b4ce033b0051e6",
    "round_03/fit_000/validation_metrics.json": "5cc16664c462cbd711d3c61990c19cdf117458f0c55c49bafa35ad0483580586",
    "round_03/authoritative_packet.json": "1180c495a1a3b44e21a7bcd65b018a39df312d60fe58cdc2653ae7b7e2666f0b",
}
RELAY_6101_WAVE07_PROOF = {
    "logical_index": 0, "payload_sha256": "ee08e66f7541e5081b341ae3b104b5839294dadbd37afe8a09c24200dae49ad6",
    "previous_request_id": "d800a262bd72eed1dba8cfa0d8d5d2f82b2841da7f28320be4ab24ab1707b61d",
    "previous_terminal_file_sha256": "bccac7285af180969617c3eca42b03480695f0d25f2e2aafcc7db88047f671bd",
    "previous_transport_progress_file_sha256": "bdb1e5376d1223808aa8df262ee6308df6f779a0f4fbae8bfdfb326812cf5e78",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795700386.0270398 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 33, "maximum_logical_requests": 97, "maximum_physical_attempts": 193},
}
RELAY_6101_WAVE07_PRIOR_ATTEMPT = {
    "attempt": 0, "attempt_id": "290284ba893c42f79a67a9b5bf88159b",
    "job_sha256": "10768096f58639ab0558274872135f1b412983a80531c74ca3ad7d88529cf1f1",
    "result_sha256": "0ab6189a6f4dd5e1c2dc3986701aaeb20d681aa51a9b867a4a62bd387742acda",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE07_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_text.delta", "sse_event_count": 1092, "last_sse_seconds": 1930.1032520001754,
    "provider_request_id": "135efa95-4f78-4992-930c-2d2aefe18baf", "model": "gpt-6.2-astra",
    "response_id": "resp_058c376bd0a872b0016b082ba0164887d2a9347a98d0dca4c4",
}
RELAY_6101_WAVE08_CONTRACT = "d25e49bde420072e42f30871dbd08b66b18330e5c9c3c74b6d14de54f43cc95c"
RELAY_6101_WAVE08_AUTH = "a03d6c829065244f62c87e3628bf54c3ae80e4dd420db3b02cb7c44aeeb70837"
RELAY_6101_WAVE08_PROOF = {
    **RELAY_6101_WAVE07_PROOF,
    "previous_request_id": "2b2b9d5ba368ff9e49e4dbc610bed027fbbad49039ca732604a36314758fd177",
    "previous_terminal_file_sha256": "a774e46c0f0989967d750a3756a85e639822a11ba29acb67e2ad4448e1a0a383",
    "previous_transport_progress_file_sha256": "070f12991d2f5b8980c5bb944ac54971bdc4cfae4dd1191b8147a4f1188846ca",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795704665.0379763 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {**RELAY_6101_WAVE07_PROOF["execution_budget"], "maximum_physical_attempts": 194},
}
RELAY_6101_WAVE07_ATTEMPT = {
    "attempt": 0, "attempt_id": "e674f38eefbb4e9fa50824f61f932400",
    "job_sha256": "fa409a947fd079c30940a83194672613070eb2aa500ba1071df77759566e743a",
    "result_sha256": "f3b0f8be705e9914600ace137fe9bce9fa036189cb2c65eb6a6866c7b1b74a98",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE08_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta", "sse_event_count": 648, "last_sse_seconds": 3314.1049271670636,
    "provider_request_id": "16af7ace-dc40-4203-8032-039d98bfae0b", "model": "gpt-6.2-astra",
    "response_id": "resp_0af34571a9ede41c016b0839433f6087d2afc55981c5f34a34",
}
CACHED_R4_6101_CONTRACT = "bb5be260280e3c5c267bad90ed1af0145e57f791e866c670a34311616bd78a04"
CACHED_R4_6101_AUTH = "a01c0466db9c147861e8d644bce946e7cb4a29b29610bdb3818c6b779a32e6c3"
CACHED_R4_6101_MODULE = "run_v13_6101_cached_r4"
CACHED_R4_6101_SOURCE = "eaca87654d2430aadbb1880aa9f2ac06d7c217367bf2f658688ec30f7147966a"
CACHED_R4_6101_FITTED_CHECKPOINT = "f36b76ecc00dbef319b4c9f6489a57aae861ae0834ce8b3c5178cb6872694294"
CACHED_R4_6101_LOCAL_TERMINAL = "0bc22fc8b112e177c020abb88a52d83bd9e20e7491d49dc275c41ccdfbd9a460"
CACHED_R4_6101_TRANSITION = "ba0f660256667b1ed7eda8e224b78d911149857252da9aa1641c3370b8d97dc7"
CACHED_R4_6101_PREVIOUS_CHECKPOINT = "1b8181b779824faa4f292e69889cc55d98e2f4f83b0124469349282f42b9bdd6"
CACHED_R4_6101_MIRRORS_SHA = "9e4e35ff3080066c185268f9f3adc5e752ed72c7eeae8484e8185c96e45a04a8"
CACHED_R4_6101_BUDGET = {
    "first_logical_requests": 1,
    "maximum_logical_requests": 1,
    "maximum_physical_attempts": 0,
}
CACHED_R4_6101_REVIEW = {
    "provider_post": False,
    "attempt_id": "c4ac95fc97414c9aa78ce5e9ffe8d4b9",
    "request_id": "611835b9380be5195601e09bf90efbbc131fdfa04b241ac514031d92c453738d",
    "job_sha256": "e958c64ad2ad557c35cd0ed53abfe7b1f1cb9921bf6ed7c9da0afc5101a098bc",
    "result_sha256": "4d1f7cd08e6283bb10964f30fa49c3bd61c7a4899f03ba7693934ed20b2fe834",
    "raw_response_sha256": "d4f7267158fbb7acda8b048003d7e6096e366ebb0fbbdef8f211af168056bb7e",
    "response_id": "resp_0c7893dbeb805928016b084b21d93087d29a9ce0051335c26c",
    "final_text_sha256": "71bc2bfaf31f9fc318c2dac716d6afff24d0318099872d51c555b686a7ebf5c2",
    "original_plan_sha256": "880493450e756eb1487654b70df20b03a8f93e098297f6698ea7e688c736b9ca",
    "normalized_plan_sha256": "ac348f5ed60165e9aa7c0a8589be4a1dbe2a69644694b8261c0a4b00418f2d41",
    "ordered_candidate_ids_sha256": "089314d92777b889207fd3e1fb2a42a12c049ea66834ded095fdf8c3e8582946",
    "ordered_reasons_sha256": "1f381544255cafc82faf366407eb315c472cc6a2f6d8e1aa10a635ca412a2049",
    "frozen_order_sha256": "27781e476a036022f8de5816008c94c9b3ceefe7aa611e0e18417728b890349d",
    "archived_terminal_file_sha256": "279c2cba1a8e758d1fabf9afdf282e0ef2f274eac879917d9b4cda94c68f8931",
    "archived_decision_file_sha256": "e2318b2ce793487b7149a9577f815a1e8610d74de93b90f96dad975225b06bbb",
    "previous_seed_pause_file_sha256": "f888e524590442f261821b7156caf1de057401b329993a189cc8198d48ea567a",
}
CACHED_R4_6101_FAILED_CORRECTION = {
    "attempt": 0,
    "attempt_id": "c56c1f108e2e44a99c84c061251974ff",
    "job_sha256": "c0b5f2df29d3dd2fa83687d0b3c593b3ef012a438a8eb672bbef749de015c5ff",
    "result_sha256": "a1d7732438ffcd116e6a90c62590ca880cceb9de231d900abac938943624baaa",
    "error": "connection_interrupted",
    "provider_dispatched": True,
    "ambiguous": True,
    "request_id": "91d3d8cdae7459a30d7edcb1a0a980db64e67a62e3ae8eb8f5be56a0254ab311",
    "terminal_file_sha256": "108438e95b58118d79032ecc4e6906b2be5866c1a8e77925094ae421b53c3f7f",
    "progress_file_sha256": "37346ba002baa871cacf381c005ef497ec25b373dd3b47051e32dbc306cd590f",
}
CACHED_R4_6101_BOUNDARY_FILES = {
    "round_03/memory.json": "2953230055565dfa2b15e5542b228c4f6df6f2e05667aa252cce3eef49d03f9f",
    "round_04/source.json": "4ac4de887ab7b56d894389a2b8a087a32ddc450ab878331a19da3d1ec2031b47",
    "round_04/transition.json": "ec45b661e47a89c3d04123732c0d897c68fdc938c1ed167f2476dfc6fd2697e4",
    "round_04/authoritative_packet.json": "92755d8994db1301620dcf002e588dbfe9d3d3a49257f0878659330269926db2",
}
CONTINUATION_6101_R5_CONTRACT = "e061a3321283a52928bc97de3f3f704abdc91aad435a79ca8f374f36af4c92f0"
CONTINUATION_6101_R5_AUTH = "7ed54929fa4513e5b4b9060705b5d4cd582094c6f225b4e1cdd8e1056b60a8e0"
CONTINUATION_6101_R5_BUDGET = {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
CONTINUATION_6101_R5_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_6101_r5_continuation.py":
        "4a95c0ab3a42c4ae97886f95b27c49db9a8997353ca90c8a41214eae3d8877fc",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_6101_r5_continuation.py":
        "711467d62f579b3b6894f01e1117d012327d5360ec4700784b0746496a0ede79",
}
CONTINUATION_6101_R5_HANDOFF = {
    "cached_selection_file_sha256": "39581821fb07c496c2427b4777525834e55e02419f94754b24f9ba8c1e854b06",
    "extension_source_sha256": "ff67b9cd13896991283e9491fd5856b0fe88ae31cff4067d40da46b4503bfb94",
    "extension_fit_audit_sha256": "62ac68dd42b0427c4ff4bbc07d7e26683ca379ccba7908adab992156a71528a7",
    "extension_freeze_sha256": "c5698b89d0b6f29d41330e8fa162aa519826135e3e877fbab5e1afa2be4dd652",
}
RELAY_2357_WAVE01_CONTRACT = "3c20dad3f4c772011e82d8684e7bfe4204a988e32f59b26b90390507ad6bb1d8"
RELAY_2357_WAVE01_AUTH = "5a507f2471b94d81b3eaae590d6c027bc2e8d3369a424550af0def0aea8c25f8"
RELAY_2357_FITTED_PREFIX = {
    "round_04/source.json": "2506c383d9e1a2de8e27635a36a4198121cc6772ad917685ecbdcbd31b605cc9",
    "round_04/boundary_memory.json": "e70a0d42f6aa9b94fc73046408bf4e2ce9d3082d896f5993909b29f3ca0d5620",
}
RELAY_2357_WAVE01_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "bde86b9f77d44228bc6d1d2ee34287e4",
     "job_sha256": "e39f6aaa75b557ad92a973bdd9a2b85be915d1de98c3b07d8b0e0e574e9ba78f",
     "result_sha256": "940cf534df4872eee6cbb51367170d9f882e0c59e0f3e4bfb7da1c0764aa2ab7",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "29e6b83977b14450904542b81b5efbd2",
     "job_sha256": "8cea52ac8526860bf3a297c691238295482304125889bfef518cf1f89c243ead",
     "result_sha256": "14b2e29fe47a84bd1c194d11f0adcb39f35f8283eb89a73d93a822117dd1fd76",
     "error": "timeout", "provider_dispatched": True, "ambiguous": True},
)
RELAY_2357_WAVE01_TERMINALS = (
    "4c50f3cf04f5dab96cd254ebe18e6d17e2c98d45f9dd35a38ff90eeeceafedc5",
    "984f751b7e162ab3b73efec0d5637884c8af8df1b75a6122c7e7068f8ff9a198",
)
RELAY_2357_WAVE01_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "3e6e8f5986a2dc8ea161d69ee967490341fbdcfb05bc8d8949c056d9a74ced77",
    "attempt_00.replay.terminal.json": "4d6e133baff411b80f58ff3a7ea15724f6ef8c40cc4a6616907e81a2ba08de70",
    "attempt_01.replay_link.json": "1d865c07dd2f269ad17f8b5991a3e0be6a1c14b3f5645fa8da1722b994a587cb",
}
RELAY_2357_WAVE02_PROOF = {
    "logical_index": 0,
    "payload_sha256": "302fe66d28c6c9aa4cfd604a56d623cc0f6d7a8d4bd53f4ee67e194732bf39ea",
    "previous_request_id": "0c88dc80f734af1eee98a8d01d5992725e8a2ce68787a1a572150e714beb71b9",
    "previous_terminal_file_sha256": RELAY_2357_WAVE01_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "2f7e4ac250c929d338b61c3ad8f25a3ee678900468649a1f041f5ed8c08368b6",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795687279.5865693 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187},
}
RELAY_2357_WAVE02_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline", "sse_event_count": 5325,
    "event_type": "response.output_text.delta", "last_sse_seconds": 2957.5922856668476,
    "provider_request_id": "48fda888-11da-47f5-9c99-222688c89689",
    "response_id": "resp_096ae4c1a60efb8b016b07f5cf8e7887d1b0e15f434504077f",
}
RELAY_2357_WAVE02_CONTRACT = "035481e8d0864a48445d4a604497bd9d934d21b1b6f6492c228222e3f606b5a5"
RELAY_2357_WAVE02_AUTH = "d2e03688bf51035fdb24f73e0d1c7dfe39a3306935a528a9bf2000fc42d01fd7"
RELAY_2357_WAVE03_PROOF = {
    **RELAY_2357_WAVE02_PROOF,
    "previous_request_id": "58764c4b906283071162ec91fca7e918ef30268454eb2ec804fcbec7b224330e",
    "previous_terminal_file_sha256": "4c9f3ba23256ec6798f73e3ef54daa4eaa70bc99accb63d428cb329fe9bc0450",
    "previous_transport_progress_file_sha256": "b9d4d97dc0a314a9406930830f389f9dde3960ff3ac26329fc058c4d88fb2dcc",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795688074.0102787 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188},
}
RELAY_2357_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "8061ec9fd687497ebac1ef472bc086d3",
    "job_sha256": "8a657b0c0ecfc0ae8b2f8656849bc8492b0b225695d84f8935fe27c9084749f4",
    "result_sha256": "130c86c96e74c6476716d522bebc048364ea4f396cb404885c3d799ba204630f",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_2357_WAVE03_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "2efb8fb0-cffc-49db-bcb2-18794136153c",
    "response_id": "resp_05dfcfb75072f710016b0805ba45a087d0ae49da3cdad433f8",
    "conflicting_response_id": "resp_2efb8fb0cffc49dbbcb218794136153c",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_2357_WAVE03_CONTRACT = "6adf6cdae83afb6ca89b6e2b04483f1b1436cfe7188579f3b250ee6548818c3c"
RELAY_2357_WAVE03_AUTH = "985d25fec830d958301ae43b786bf678590a38c890971e992216a0dc543026f9"
RELAY_2357_WAVE03_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "86d12e7086c14d0aadd056d318ce3ebb",
     "job_sha256": "6973100ab086b43dedaddcd5e649a574acf1c77acf531bdb901370171a425084",
     "result_sha256": "e4e45a46db36440cf600def9745a19f0d244124d00ddcec17427f07da1a0a4e2",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "c7b932c208cd4e6f83d0879ef7c1878b",
     "job_sha256": "ff68e1fdfa784741d52f4df56a79cc350303b9b6e6174863a8a49c059e18cc60",
     "result_sha256": "a548417ce201488b9ca203e8b02edc4c179ec0c0e4233adc90683056e82ebb90",
     "error": "timeout", "provider_dispatched": True, "ambiguous": True},
)
RELAY_2357_WAVE03_TERMINALS = (
    "91db59ce07f81d97fd6dd317614f188d7457422d43e0b773a5b8b39da12fb427",
    "09858195dcb9316f86f45d19d1ac8c5eecd9e7346f4e0541e64e71ad1be40661",
)
RELAY_2357_WAVE03_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "b38e8214726b1296a5c0c27f2d0d60ff9daed00575a2bef2fffeff270b291a8a",
    "attempt_00.replay.terminal.json": "9546fb7633e5944238ce86d87d73b5e4a32a0a3d58f94b51af9a5821fa61d431",
    "attempt_01.replay_link.json": "808a709c47b4ff8e435eaa7559754c206aa81d55ce67adec6237adb97f8bf901",
}
RELAY_2357_WAVE04_PROOF = {
    **RELAY_2357_WAVE03_PROOF,
    "previous_request_id": "68deaab12dd523312fcafd7eaeb27ec0589e876157b964ebfb52e6caf29bd417",
    "previous_terminal_file_sha256": RELAY_2357_WAVE03_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "3bf15dcf933a56dcf5788a55559b0f23b47021d300bfa3cd2acf416566208257",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795690463.3580766 + RELAY_RETRY_SECONDS,
}
RELAY_2357_WAVE04_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout", "sse_event_count": 14,
    "event_type": "response.output_item.done", "last_sse_seconds": 289.73664454207756, "model": "gpt-6.2-astra",
    "provider_request_id": "87a5a63a-102a-4c9a-8e13-c5c93cfa0ca0",
    "response_id": "resp_086e7bc1ad0aba55016b080b4527bc87d0bc905983bc86a586",
}
RELAY_2357_WAVE04_CONTRACT = "fc19d508b0374d79fb7a015f50b68c606afaf4811c3c87fbaeaa216699759f47"
RELAY_2357_WAVE04_AUTH = "dc83d4755b0859fd1a7db4989f23541a6cd552f1418ea5745aa27ad2dff361fd"
RELAY_2357_WAVE05_CONTRACT = "0ea96d7929e9bdc395bd7bbf05ccdeaebceb5805c5ba73aa8f44985635d57967"
RELAY_2357_WAVE05_AUTH = "ec14cc82c953da5ca45361aa2abec38f35a88d3b7319504f63941536e40db20c"
RELAY_2357_WAVE05_SOURCES = {
    **RELAY_RETRY_QUEUE_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/provider.py":
        "9b77e82920a303b9429957372d0943a1c729fecca3c80c00d86e0411a1c74bd0",
    "studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py":
        "25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739",
}
RELAY_2357_WAVE05_PROOF = {
    **RELAY_2357_WAVE02_PROOF,
    "previous_request_id": "3aea456efc01f2a7a0d4b82e1c0b7b3a8bf1270079dfa7065811c33ae1b18d8f",
    "previous_terminal_file_sha256": "5f48a0457c89a6bfa32b0353993d3228d920faeef7d27e4089e792ad09ea5ec0",
    "previous_transport_progress_file_sha256": "17f568c466f3bfdd75e23f4082ab57c599007bce939db3231d5ad0c7b63387a0",
    "relay_not_before_unix": 1795694625.301756 + RELAY_RETRY_SECONDS,
}
RELAY_2357_WAVE04_ATTEMPT = {
    "attempt": 0, "attempt_id": "fff7561203064189b1f10fdd96e26e61",
    "job_sha256": "685292e3160694d3cd861dab060c2247668d36aff62279e5a0f6df2c12853e87",
    "result_sha256": "9e09937ddd425f157988ffeee0389317c50e388c583a49aea7f106c65fdd7b6c",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_2357_WAVE05_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta", "sse_event_count": 3170, "last_sse_seconds": 3181.4656675001606,
    "provider_request_id": "df645dc3-e434-4b35-a26b-14b507b18be4", "model": "gpt-6.2-astra",
    "response_id": "resp_05749099683d24b3016b0812a205dc87d090ff8bd9db36de56",
}
RELAY_2357_WAVE06_CONTRACT = "be3a0f8b24263bfec2b336a21d584ca70e65a578604f7c9d6e3b4085c7431d5c"
RELAY_2357_WAVE06_AUTH = "c3e3390acc7c3e6bf0b12105d79b6c87c4429b789fa72d1de4ac5e2920841eb4"
RELAY_2357_WAVE05_ATTEMPT = {
    "attempt": 0, "attempt_id": "a6b141a821d649e5bc2a33ec5cab1638",
    "job_sha256": "411bd92256e1c1cf38498e48dc1fcb655af4d69070402034af0fea9d2afd8654",
    "result_sha256": "42399347447ba316b457e8b8df2c0ed48913ad03c4e864ff61597a13e23cc477",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_2357_WAVE06_PROOF = {
    **RELAY_2357_WAVE05_PROOF,
    "previous_request_id": "d1003c4a55978f13c25302a0e22e335e6b3e0553809c65931ec6874b102ac4e6",
    "previous_terminal_file_sha256": "62945fd3e4bb2a9c4765cb9a15fce22cd2f2687748a42e55694f3efc5bb6427c",
    "previous_transport_progress_file_sha256": "beb8b632fc743457b12c9d1328d37806cd739d849337b93ebfb8159ffa4599e1",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795697966.9968047 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188},
}
RELAY_2357_WAVE06_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_text.delta", "sse_event_count": 831, "last_sse_seconds": 1604.6706121659372,
    "provider_request_id": "b1a6449c-09ec-4b05-a253-aad6e5207e9f", "model": "gpt-6.2-astra",
    "response_id": "resp_0dbb7f15ca0e4a12016b08237b136887d293cf127a7e3cd4bb",
}
RELAY_2357_WAVE07_CONTRACT = "9982670e76d85b98ee971ac612c2f9126463946b7712c3cb2ff9802b1fae077b"
RELAY_2357_WAVE07_AUTH = "d2088d434902bcfd7a98df850f2757e88654788d656e4f88aa18a373a939a62e"
RELAY_2357_WAVE06_ATTEMPT = {
    "attempt": 0, "attempt_id": "46684da6118f47f5b6144ac8d60c11ad",
    "job_sha256": "1ac263c754aaa0f5a44200f2815e4d1e8a17b1806e05f5600ab1ec5dae74d94a",
    "result_sha256": "2ae5aea87b849a8d3ddd0cbc908855a348a574ac1bc2d6f9aef6e892a6e800b4",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_2357_WAVE07_PROOF = {
    **RELAY_2357_WAVE05_PROOF,
    "previous_request_id": "02be42da96eb22bbe91aa1c6b9288c13303f964330728e6b1ed0d843d898f140",
    "previous_terminal_file_sha256": "691773641afc17c78915325644fc2918b45571d02a63e742566e9604415930dc",
    "previous_transport_progress_file_sha256": "5b44830f2e69aa1585764e233a89f6b4d786e8ef4a634915bf12c1f3e719c2b6",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795700982.1268728 + RELAY_RETRY_SECONDS,
}
RELAY_2357_WAVE07_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_text.delta", "sse_event_count": 440, "last_sse_seconds": 991.6895799168851,
    "provider_request_id": None, "model": "gpt-6.2-astra",
    "response_id": "resp_0d6c9b78cc72e853016b0831a918dc87d18fca9c2e4227ca18",
}
RELAY_2357_WAVE08_CONTRACT = "56abeeb8d3c4ad38c286a3b258a34298659f5596a8ee6bd970473d5e450c06ad"
RELAY_2357_WAVE08_AUTH = "56127c93850689717abf865b3b856e07d73a26b9779d608d673121988f8990e9"
RELAY_2357_WAVE08_CHECKPOINT = "b31f06a0aa8efeccc08d6a6b709ab7f7360acfac56b9041cccae34c1363be501"
RELAY_2357_WAVE08_FITTED_FILES = {
    "boundary.json": "15f0871e497fb397fa8b9088ccb8dede87e7b42812ed18471b79385d4b685d20",
    "round_04/selection.json": "418c30b287d441f4cac20717bd3bc341bcf6cb35382634e6d8263c57ef142dca",
    "round_04/batch_freeze.json": "54660ed0cb3910c18f5215b195ba3691aa091283b47e28a8e9124962152e3da0",
    "round_04/memory.json": "ca68e2011ba224b51a5d718fbdda208aa64896fa606bcd484623e44aec9823d8",
    "round_05/source.json": "a60e0c92322ffb2a7f40fc9e5a20894aa6905e5048eaf88520088e08c5b25d00",
    "round_05/transition.json": "bf855d43455d4ae2e1d2ff7d50a17b94be5c1adff2782c47fa5011d12f84703a",
    "round_05/fit_000/fit_audit.json": "4e89b08f06578ea4d901a062403667ab3508787629b93b70ccc5ecdd7b69aa80",
    "round_05/fit_000/validation_metrics.json": "f6397e258dcaf052022b207166e93ddeb677ce02fa02a9baacecfd0e537f20eb",
    "round_05/authoritative_packet.json": "7b6ba694100cf4b61f17e5a69a50565e8ad8b290d8d45f068478af2b5dcfe101",
}
RELAY_2357_WAVE08_PROOF = {
    "logical_index": 0,
    "payload_sha256": "9bbfd69fb3e88c99e81a9ae6d12df1df4488c8a4466bd2c86e925ec8f041db28",
    "previous_request_id": "6858cfa565da51909475a563cd8f8dd8ec02b7191b16009f521365af28d42a56",
    "previous_terminal_file_sha256": "33af82fe12d27779bc805489d407cefb2cf23278d2284c70d98faaf0abac7394",
    "previous_transport_progress_file_sha256": "068aa3de0d96fe4f4052c3e164adcc2f7cd95fbed2b2a466a3955e7db42c4691",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795705060.1421165,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE07_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "2c4159b1abb44af2abe95938b6f5014d",
        "job_sha256": "e356e8e375891801a51898530271468ae6da6afd44e4c0f4b4d6d9602e1d293a",
        "result_sha256": "6ed590d163f7915c8f43699fac87df073216e778d1daab129a4b1390a12ea918",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "80c5ef6e55cc497891e88860772de966",
        "job_sha256": "3982bbff222cd93ddf924579a63e58f18d769a838d6f1e38c847ecae86745909",
        "result_sha256": "da5ac22f12d9517cb611246fad8f575d44200b993f93d5e2154dc14767f14f9c",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE07_TERMINALS = (
    "eb7fdd44e5d569638f5a6b54eaf5127ed6530660b4f1fcef63cefedd48ccce55",
    "33af82fe12d27779bc805489d407cefb2cf23278d2284c70d98faaf0abac7394",
)
RELAY_2357_WAVE07_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "7a4036560f2d270a3a2077dec3f44c27f19ce8dbad7e15e9acb11fec54a2a031",
    "attempt_00.replay.terminal.json": "fb62b42b673fd39235d6953e0f61574d4f40e96e45d3ab80c5e210fa5504d5e5",
    "attempt_01.replay_link.json": "969c43ad504e1c4bb2641ef163f0fa757513bcd0447a9d9e4cee144d69acc892",
}
RELAY_2357_WAVE08_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "22215ed1-3cc1-4de6-83e7-300e9681da73",
    "model": "gpt-6.2-astra",
    "response_id": "resp_08301c347a6f76ee016b084759604c87d0abb04a7ee20a94b8",
    "conflicting_response_id": "resp_22215ed13cc14de683e7300e9681da73",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}
RELAY_2357_WAVE09_CONTRACT = "f85aa65e0caeb48b419927c074278f0b97ee483c66180c06eab2e2588aa85f77"
RELAY_2357_WAVE09_AUTH = "fd640132095e581fbab5377114ceff22f06d9931f30adb5bd9b9c6b31a9df663"
RELAY_2357_WAVE09_PROOF = {
    **RELAY_2357_WAVE08_PROOF,
    "previous_request_id": "4e74466519da52c01150e90ec5407401b9b890bf016b88c64ed61095da9af971",
    "previous_terminal_file_sha256": "d9ff8d5817c358605ff5d6e7fb3d2df1e2d9c12e3346fa5eef25997880432987",
    "previous_transport_progress_file_sha256": "184e2ac31b3c9abfa6ab43290d46207eef16888366fdc653a46bcdc7e9949115",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795706668.8023124,
}
RELAY_2357_WAVE08_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "dd59937e8b7a49a085984f24ee0effb7",
        "job_sha256": "27cd276c4ee00893de0c177dde6ddb8d0bd9854bb092d3af761a5ff1c00195d2",
        "result_sha256": "d9cf7e90039b6e994ce800d71db2cd3aea812601412cfc2c87d6ae8bc5e64169",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "f4b70a1204d74013a247a4d8653ec3e9",
        "job_sha256": "a39c2a0d58719bb6441da89c1eb641815e24f1a3ace0e4e8df39ccf4803aba6d",
        "result_sha256": "c85d5f55acef98c9b571e82abe97fc4b544eb557f08a8a11b27a6194262c3ca9",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE08_TERMINALS = (
    "262bdbf16241edb3f02344ec071637a5aac806a03012a111a8a66ce2bef2df40",
    "d9ff8d5817c358605ff5d6e7fb3d2df1e2d9c12e3346fa5eef25997880432987",
)
RELAY_2357_WAVE08_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "2290e6d843b58e45aa5c266c49f994258261eee5a9c1dd7830b0f890f263dc9e",
    "attempt_00.replay.terminal.json": "e5cc4ee4ad93c3cf6015e4f49293c1fab3012c7c0760bcae96755c8cda92fbd0",
    "attempt_01.replay_link.json": "33b610278022a90ddea6d0ab36f042fed266d1a31611c70fee05bef7d4e6b1e1",
}
RELAY_2357_WAVE09_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "48ad4b49-24e9-4e69-97ed-3f8dc51d1ade",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0fff34fcd9c0ee76016b084da47b5087d0887fa5dc3059c3a7",
    "conflicting_response_id": "resp_48ad4b4924e94e6997ed3f8dc51d1ade",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}
RELAY_2357_WAVE10_CONTRACT = "fd02d26a0e700633a21bc4f26214be9393d76fad1a852d4482262371b1b8959f"
RELAY_2357_WAVE10_AUTH = "779a4a91f62c6686dd6979c14624e3bccbf7522f895431e06685eeca21525388"
RELAY_2357_WAVE10_PROOF = {
    **RELAY_2357_WAVE09_PROOF,
    "previous_request_id": "cdcdf87d1d2ccd568decbe711b4ecda2c3dcf2f7d86620cf603c0d97ff304b65",
    "previous_terminal_file_sha256": "2653d4924e63d5999b2f78fb6b860460d820b7a730395a9dfc6352588f355d14",
    "previous_transport_progress_file_sha256": "909a5d5e6d3654dfc7259ac8bdd39a0d07085293162ef05e2b7b262aae148f3a",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795709663.170326,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181},
}
RELAY_2357_WAVE09_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "c75a2df1d54c4abeb9c11e242dbde066",
    "job_sha256": "b581e90bd4f97439462554ed51aa58de9b5b993022b3ac1cc7f6037e23d009c7",
    "result_sha256": "5624186190f2ea4cb79f2053e53c1a02b873708dc33b334cb645d51ee983fa39",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_2357_WAVE10_OBSERVATION = {
    "phase": "transport_failure",
    "error_type": "ReadTimeout",
    "elapsed_seconds": 1828.1092890840955,
    "network_phases": {
        "direct_ipv4_preflight": {
            "duration_seconds": 14.24446979095228,
            "observed_seconds": 14.263882874976844,
            "state": "complete",
        },
        "direct_ipv4_preflight_1": {
            "duration_seconds": 14.24446979095228,
            "observed_seconds": 14.261098542017862,
            "state": "complete",
        },
        "direct_ipv4_preflight_1_dns_tcp": {
            "duration_seconds": 1.6504104998894036,
            "observed_seconds": 1.6687907089944929,
            "state": "complete",
        },
        "direct_ipv4_preflight_1_tls": {
            "duration_seconds": 2.2314766249619424,
            "observed_seconds": 3.903982625110075,
            "state": "complete",
        },
        "receive_headers": {
            "duration_seconds": 1800.0037664161064,
            "observed_seconds": 1828.0736891250126,
            "state": "failed",
        },
        "send_body": {
            "duration_seconds": 10.512595750158653,
            "observed_seconds": 28.06884575006552,
            "state": "complete",
        },
        "send_headers": {
            "duration_seconds": 0.00021974998526275158,
            "observed_seconds": 17.555847208946943,
            "state": "complete",
        },
    },
}
RELAY_2357_WAVE11_CONTRACT = "e51b23c0ce9271c6c506fc0bb10510e972f5821f4741c793385b8aa014748ce6"
RELAY_2357_WAVE11_AUTH = "91f1d4e2238a5189ff2cdc3ae96607dfcea2346f17ed7d98c114f92104b473f9"
RELAY_2357_WAVE11_PROOF = {
    **RELAY_2357_WAVE10_PROOF,
    "previous_request_id": "261148011c2e9354de302ebbd9323dae15536a1130677ea9b79244bb601a0d20",
    "previous_terminal_file_sha256": "28c5247df4511c6d2f8b58efc48322f3eafcc6e492ed1bb1a2ff01d002151f18",
    "previous_transport_progress_file_sha256": "df8b5fd031ffc15dc39538d9fa88cfdcdd7cba664468dbe7bf2558236dc139e7",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795711758.937361,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE10_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "d2a8dbcdfbf14301b3e91b626cdfa716",
    "job_sha256": "c8de05a1c0d4409c6cb8a89341b0e12def7d356a2685057f52c81e928766a4fd",
    "result_sha256": "377af21c447cbb62d61d5c1a9153adfba7ddfc5783f4605de4e84790343e2689",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_2357_WAVE11_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.added",
    "sse_event_count": 17,
    "last_sse_seconds": 501.662876291899,
    "elapsed_seconds": 1401.6690100000706,
    "provider_request_id": "7b25cdeb-4598-44ba-98fb-b5a3c924696f",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0bc4c24a3c1b9d40016b085cfec74887d28fa81ed2a6343b27",
}
RELAY_2357_WAVE12_CONTRACT = "b78ed3a580e8c22077b87818c988b83e30d05a41bb93971ee8f65bbe35af8f60"
RELAY_2357_WAVE12_AUTH = "60dcf56a1b35c79b3971bf6fa94e1d7809326509d479cae5be49cff4454d286b"
RELAY_2357_WAVE12_PROOF = {
    **RELAY_2357_WAVE11_PROOF,
    "previous_request_id": "c734d0704db6d98ff61321c2ceae8a0e06100c976cd681662b0e96fec98d38ee",
    "previous_terminal_file_sha256": "652f819e1f820de798db4ee803a6b34508ca26a4970f080471858c6372ffc47a",
    "previous_transport_progress_file_sha256": "ad27ef2a28d57adc22b7ac1a50cf57dfb511ef3df6c51ffee9940f5b5642c42a",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795713361.660755,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE11_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "67bb14f783184999a8ba51aecb620cad",
        "job_sha256": "56e9a3b53001938eaa18d4cc489f36854c34f43723b966a70b47abcac57ff08f",
        "result_sha256": "61786745aeae39d1a1689828b47373f88250f6f2b99ae878eda51734195c77e6",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "5ee7f72f23d3432db2cb674ee0c809e3",
        "job_sha256": "ff996b9c94bc01ca7f563a9c43f436403eb7d99e3f99818fa6cc7b370b5407cc",
        "result_sha256": "b60125bd3428d75e95c88c06f0da44acf7ac3175b678c119464019b7af7b3ffc",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE11_TERMINALS = (
    "e0f45cde999a2402b61a3fd80b70d5ea0835979d4ddff2f709c01a51afb58123",
    "652f819e1f820de798db4ee803a6b34508ca26a4970f080471858c6372ffc47a",
)
RELAY_2357_WAVE11_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "95475c02ab29c825ea18ff55690ebdb428a343ad60861618c44a9a6f07f7f1d3",
    "attempt_00.replay.terminal.json": "b8e7925c3f608af1045608445f2688a1781d45e4d4722317be8dc1d3b46e6134",
    "attempt_01.replay_link.json": "5717bfca88e6f0297cff08de37a963734045718a50fe705f4f801e95837f8d6f",
}
RELAY_2357_WAVE12_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "6e90dbbd-eaaa-48e2-a396-2d6fcbb1219a",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0701b3cd45663007016b0867c08f5c87d184be9b44d19aa5a3",
    "conflicting_response_id": "resp_6e90dbbdeaaa48e2a3962d6fcbb1219a",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}
RELAY_2357_WAVE12_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_2357_WAVE13_CONTRACT = "971509317d68b4d787bb09d8f41e6d0d3dd7d2dad86e446a8b1aa1be3881a578"
RELAY_2357_WAVE13_AUTH = "b885108529b6475fa90ad880d4bd28565951fbd48f48b08746373db6038086ea"
RELAY_2357_WAVE13_PROOF = {
    **RELAY_2357_WAVE12_PROOF,
    "previous_request_id": "7e4eaa2366d785ac178e03befa08cf5582eae475865a33a27d9c17fd518e1b65",
    "previous_terminal_file_sha256": "19fd3b81b8ccdce193cb59ae5d4c1edca3f356bd8b3de739b3688cafcd6684e2",
    "previous_transport_progress_file_sha256": "5c341138e6cf4e2eaaa990cdc7a518b54164a2176a1ef0b247676772450a584b",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795714691.0751874,
}
RELAY_2357_WAVE12_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "60096c8a5cf849d6a5c3a06bd203691b",
        "job_sha256": "f4662af19613dded4e1be6f2064320c90239c34e07497b2480eb3b2d4a98cbed",
        "result_sha256": "19b91ff37a2cb42fc31b29f662e2d89210098816fc261a65d3b3d279eeaa4a4a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "938a1fbd8a66479f87505a2824ba1c45",
        "job_sha256": "269c709de787770924539b40516ac3840a5ab09efcd9fffbf23b58e7cb6ca1ae",
        "result_sha256": "e069044a1b0da28194f44205ffc8a796d1e1917e3104d7dc23c6b5220f9d7d91",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE12_TERMINALS = (
    "6dc8d03507b5f8b8e8c87969a7a67997d7d6a8bdc14243113f6228b55444d658",
    "19fd3b81b8ccdce193cb59ae5d4c1edca3f356bd8b3de739b3688cafcd6684e2",
)
RELAY_2357_WAVE12_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "51e39e4bed4fb1c52d889d6264e2a6b3b339f0bf48f31cc02b35a2ba4a22ed18",
    "attempt_00.replay_authorization.json": "70682c9eaded0a16258580f17f2e24d29381504f061e494456607fcd8c708cb4",
    "attempt_01.replay_link.json": "8352325daa3f308d30727c140285464dc4ab68dda2cebc8714169505e595cb25",
}
RELAY_2357_WAVE13_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 23,
    "provider_request_id": "61398b0f-99ec-49ed-852b-50749894b751",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0414566f3ddd6909016b086cfea8e887d1a92c9ad38fbbdd41",
    "conflicting_response_id": "resp_61398b0f99ec49ed852b50749894b751",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_2357_WAVE14_CONTRACT = "16bb17dff9acf067e8ac53500b4795c5a08bc599b2be0fc8aed6e4b4bf13ef4e"
RELAY_2357_WAVE14_AUTH = "0d377d4a8a095733404e6e6453e396bd8900d2c06a0cf9d02f43f940d8639a5e"
RELAY_2357_WAVE14_PROOF = {
    **RELAY_2357_WAVE13_PROOF,
    "previous_request_id": "42e6dd57dacdd98571a74ef264ffb6ec15a910cfc7fd4cb4a007ea32798ba815",
    "previous_terminal_file_sha256": "3ff6a01fd8e989857a6d0a62854505beea16ce569b2960fb78dd8874c6a3c926",
    "previous_transport_progress_file_sha256": "354ce997e058bf28997db4560adb2b4ee9cc27d4d52cde2502fa5190ceb3eb08",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795717386.3467476,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE13_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "4e9bf9c29a3842c6aeb16187db906b32",
        "job_sha256": "f46af29b4109dbd02959c2ae52c6bf09665f11fa42f7958162085b95dae1b6e4",
        "result_sha256": "2a4da8598d7fba845a5916bf3a9757c3f08a8af342b9b97633e1cabe2555f799",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "c7838e3483fb4e3c8217729061109eb5",
        "job_sha256": "e7588b4ebc499c3650490fab13cfc741ac864d0070bba06c79b64a59b7adb106",
        "result_sha256": "adb99ba9e070fa4b4c1fa08a3aab0b2b84bd2b67dde4ba0e8a103f0784a04010",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE13_TERMINALS = (
    "599db9bf2b092041e422b4f5e3af0c12bb133a8f1e9b1958768ed880ac43bfb3",
    "3ff6a01fd8e989857a6d0a62854505beea16ce569b2960fb78dd8874c6a3c926",
)
RELAY_2357_WAVE13_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "420c9ddf47dafe744208c97e9363b32339a6249b87e00b7eb74c9628ba038579",
    "attempt_00.replay_authorization.json": "0dea3618537cb37d373ec63f6d73e0d7a5514d8006dcd8c795819c920290af30",
    "attempt_01.replay_link.json": "6a33bded3147f04c4ba4e1d25c41645ad0668b9a233956f8f2d483a424016d2c",
}
RELAY_2357_WAVE14_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "provider_request_id": "debb12eb-8ca4-4182-b805-e508591d8f51",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0e0ee04155c2ceed016b087787aeb087d2b946c64f4a174fda",
    "conflicting_response_id": "resp_debb12eb8ca44182b805e508591d8f51",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_2357_WAVE15_CONTRACT = "128f918c7a57c055ec4aa6cb1e1e587363f5f667ece2a68ac3b6e5715d64ff62"
RELAY_2357_WAVE15_AUTH = "c071817654fc799bad48e2d0433635315f2e2d5207df65d00eb77bf43ff9622b"
RELAY_2357_WAVE15_PROOF = {
    **RELAY_2357_WAVE14_PROOF,
    "previous_request_id": "83813669549a2b0a9761c2ad32f8fd26efd76068923f814ba7aed08ce787f6ba",
    "previous_terminal_file_sha256": "b0b295186d112c9495e45452d97cfc6182e0ac711fc71349ddc90c402612c06f",
    "previous_transport_progress_file_sha256": "35c7d6025d1f19f26e35a2464a7ade46030a115f62ff5b08ac2631ee23a016b5",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795718762.0906043,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE14_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "d623a45515c943cd88a4c93633a670c4",
        "job_sha256": "2c856b8ce3d8f6e5bd6dc6b955a8b660ddeda4d556beb5d10ce36b2145620210",
        "result_sha256": "8fa25d857d5d236c805211f3d38547a689f83bc11cce783dc71fce58524ca591",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "bedbc8d4462b41859c8ed5cc4415fbda",
        "job_sha256": "0bb64f5cc47aba3f5272fe6ce7697924668dbd63f327c7c7e5746f018a731b8b",
        "result_sha256": "a1d258bf3228429b31d5a96410dca079d80078e07565066b17409464ea5be46f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE14_TERMINALS = (
    "2211e2ceba9fdb9cc17703aa0b53cd4845410f9c8432baebbeecb60ea789b767",
    "b0b295186d112c9495e45452d97cfc6182e0ac711fc71349ddc90c402612c06f",
)
RELAY_2357_WAVE14_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "0c52248723d54dd7ad196d0046598c2f93612ecf85cdccdfb901798c8b75b090",
    "attempt_00.replay_authorization.json": "5ace23c35815a5f7e9386385034582f11997ac6f734e96ba79ce07c74d523459",
    "attempt_01.replay_link.json": "811c37ed8a98533b809696e54486a7c3ad987c735eaa8dae3262068449769c42",
}
RELAY_2357_WAVE15_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "4ce87799-dd31-4af2-af55-93bba3339782",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0a14140aa36c782e016b087cdc282887d1b93327570b3de8b1",
    "conflicting_response_id": "resp_4ce87799dd314af2af5593bba3339782",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_2357_WAVE16_CONTRACT = "a1012cd98e9968430ff976bef25c470aa30ba3c75be21e8ccdf3253f1fd77a99"
RELAY_2357_WAVE16_AUTH = "e643c5a7cf8ec93ee251eb94ce181b5278e739250bba9e7d2f5067784d0c1590"
RELAY_2357_WAVE16_PROOF = {
    **RELAY_2357_WAVE15_PROOF,
    "previous_request_id": "d3cfdf7c122f2ce4b33247003d974f2fe883c4ad6effe6ec340703cfb5ef6b8a",
    "previous_terminal_file_sha256": "91497035974234aa6433cc222c6fd76231712933b8843927d0051c6a46f7ed99",
    "previous_transport_progress_file_sha256": "30fa5ce4b14c2684c03d211a117c0bcb0db629eea383e12db25c39f395e977a2",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795719784.1234941,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE15_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "8e36b8d104ca4862997060e83c02b4b9",
        "job_sha256": "788145f476d9f48f80872424b2eac52ced920ebc8717843da4470fcba61478c6",
        "result_sha256": "d4589d7016e0d3a44d8fcc59a4375e71cfd9be807784abb7674431e359daffe8",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "6b1a6ffe4864450abae7e02daed30584",
        "job_sha256": "a1e68f4d485e7c664454574e48b0195ec94fd17b81e14406c97d6cbdff4fad2b",
        "result_sha256": "e4728fe60caf902a4a944ba42a5f70f3dcbca298b4a361934614f8f6a976e966",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE15_TERMINALS = (
    "053f6d85a2873f0bc6fe564f9a344852f2c5815349702c040351da5f910a70bf",
    "91497035974234aa6433cc222c6fd76231712933b8843927d0051c6a46f7ed99",
)
RELAY_2357_WAVE15_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "d1dfd7da1195530e09083797cb188c33d6a28f32f7f94da96c86acf855144ebb",
    "attempt_00.replay_authorization.json": "5f788909ce9bb3b7499d2288b461066802163b4c1a2617a6721187ad9d218d38",
    "attempt_01.replay_link.json": "a9fb5a334645c5debd1c7854b260dfca8489bd990d419fc7ab925e986d8afd74",
}
RELAY_2357_WAVE16_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "provider_request_id": "b8c41ed7-e61e-44da-b3d5-862c071825ca",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0c05a0f34761c7f1016b0880e4c5d487d2bccf69184f1dc200",
    "conflicting_response_id": "resp_b8c41ed7e61e44dab3d5862c071825ca",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.09210079186596,
    "elapsed_seconds": 197.80961004202254,
}

RELAY_2357_WAVE17_CONTRACT = "9c5b7c8c7850af16feb75642d5ea7f29fe45a5d503e2d2ad1c65450bc4d79b3f"
RELAY_2357_WAVE17_AUTH = "a56bb51b861e1a6fb07dd52e7d140a43903cebb159c585ab325f55c301b876b9"
RELAY_2357_WAVE17_PROOF = {
    **RELAY_2357_WAVE16_PROOF,
    "previous_request_id": "c8623b69e487601a7b2cde8e4e317863e313ffccfa8a6ec8cf1e8676e77e9dd0",
    "previous_terminal_file_sha256": "b990236af5b0b825d2d6d7e197f40a45bec236e944fbe2b8319919f35e862fe0",
    "previous_transport_progress_file_sha256": "fdbf4cee9c5ba8eb79516db622e1dc2df4f267288d37877a6a6e76fa340c7035",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795722006.3807504,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE16_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "1cded70dd18448a4b1e4338f03c05c51",
        "job_sha256": "44125edd93b272f16ed1dbe67e05ed369a4491cc3b72de79f83dfa96b5f8ec0f",
        "result_sha256": "b35d28d86a0f7633fa14e16315bbef0a2d0eb8b8b78856fcd32bbe339a11afbd",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "1f7361cc146f48fbad269764e61a17b8",
        "job_sha256": "441149d2ba7c85290e5bbcda4500a5bcb4ebf8e04e6988743a4448a39267de03",
        "result_sha256": "0919f07fb7d787c4e3c472bdb275437122c276fb4071f7625264f894aecba141",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE16_TERMINALS = (
    "9e008bdaad6940eb8a612c335036c44e4369ccf227f3870ba0fe23996b1aa66f",
    "b990236af5b0b825d2d6d7e197f40a45bec236e944fbe2b8319919f35e862fe0",
)
RELAY_2357_WAVE16_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "0669996a4aa5c3d09a1d3bac854e626b45f5f5673f7823043199d55bf971eb22",
    "attempt_00.replay_authorization.json": "064979c82fa2e4a42adb9596e95d3e7e16c476a344a3b01ca6c58365120e2377",
    "attempt_01.replay_link.json": "2327fccd4d88ecc08b64f58a46a768af9fb1a3559fbd74efdacf62af7b1842a9",
}
RELAY_2357_WAVE17_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "d5147b62-a362-4996-9450-dfea6471f409",
    "response_id": "resp_00867897eadda728016b08899466f087d2b58b0392a99d2e02",
    "conflicting_response_id": "resp_d5147b62a36249969450dfea6471f409",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 198.49031345779076,
    "elapsed_seconds": 201.8770019998774,
    "sse_milestones": {"response.created": {"elapsed_seconds": 21.899365790886804,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 21.90066649997607,
                          "interval_seconds": 0.0013007090892642736},
 "response.output_item.added": {"elapsed_seconds": 21.901749040931463,
                                "interval_seconds": 0.0010825409553945065},
 "response.output_item.done": {"elapsed_seconds": 30.416553374845535,
                               "interval_seconds": 8.514804333914071}},
}

RELAY_2357_WAVE18_CONTRACT = "e6382ee8b0d441879aea0045e712183ed72ab305c046ad3320502c84f366bdc0"
RELAY_2357_WAVE18_AUTH = "82ef39e050f9306309e4b92679fdaefc481b41ab22dd60305f3d2f4fd4bafbb6"
RELAY_2357_WAVE18_PROOF = {
    **RELAY_2357_WAVE17_PROOF,
    "previous_request_id": "7c0d8d6d5f4ba90e4b52ceffd9fe27a40689aaf3b61d2c1a1d1dd0ef89d5038b",
    "previous_terminal_file_sha256": "ed30f2de9cf8e0567446f8652306f0e22551efa4ebef49ecd36a4dbc71d722ed",
    "previous_transport_progress_file_sha256": "3b44ad22c98b1c3d1ad28eb91423552d87fe482ccfba94c53e05d97a197a4f6a",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795723270.9339292,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE17_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "90186ecb2e534a0c89e883969ad487d3",
        "job_sha256": "df044a8493d99e31eef3700cb99389ded7fe753f4c603723f6e55302bb7fa8c3",
        "result_sha256": "a5a8eed6cff2cc264dd35deb0a62de2ca361b6c2be4b8dbaeb8b88b9d6d0a4ef",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "749dd3bcaf984c778355742ea5a8937e",
        "job_sha256": "be33da3381ea93ae51ca1af4e0a41c335202e971f4fe7a1117d612323be9e2bc",
        "result_sha256": "403ff997086ef987b8c4d5819b03e8cfc2d2ee505f5af13a1bce09e98ee520c9",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE17_TERMINALS = (
    "5504e96e4ad6699a39ce568b006b06c37aff849ea206b3fccea599f1f85cad62",
    "ed30f2de9cf8e0567446f8652306f0e22551efa4ebef49ecd36a4dbc71d722ed",
)
RELAY_2357_WAVE17_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "ff95e7ffbbff3cc7d42676c8fe9240db1411eb0a30637cefef606833327c26e6",
    "attempt_00.replay_authorization.json": "be92d5ec1248b306e7897b38f7238388cddc2b56ba3ccbe8e895a12bcac21e15",
    "attempt_01.replay_link.json": "ff44e05691806c1483e2abd76882961a104c8275e6c19f2750febcff6db703ec",
}
RELAY_2357_WAVE18_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 23,
    "model": "gpt-6.2-astra",
    "provider_request_id": "8ed055a3-9381-495c-9076-5aa211ebf3fa",
    "response_id": "resp_0c36d980c860b812016b088e72485c87d2adbce42f8bfd723c",
    "conflicting_response_id": "resp_8ed055a39381495c90765aa211ebf3fa",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 201.00235691596754,
    "elapsed_seconds": 215.57813533302397,
    "sse_milestones": {"response.created": {"elapsed_seconds": 35.58365466585383, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 35.58796533290297,
                          "interval_seconds": 0.004310667049139738},
 "response.output_item.added": {"elapsed_seconds": 35.591414040885866,
                                "interval_seconds": 0.0034487079828977585},
 "response.output_item.done": {"elapsed_seconds": 53.547473415965214,
                               "interval_seconds": 17.95605937507935}},
}

RELAY_2357_WAVE19_CONTRACT = "cbb985566055d926c39413abd8e111a27b99df2b69ebb09490db374c50fb403a"
RELAY_2357_WAVE19_AUTH = "9baa2c6fb711fc8ef67797c7bb83e8b4542d79c03b8973bbf102e7b42f488a94"
RELAY_2357_WAVE19_PROOF = {
    **RELAY_2357_WAVE18_PROOF,
    "previous_request_id": "6cfa51b09c6efe21d11eb3eda0568d105ed608f285b7022d2c53facf40959985",
    "previous_terminal_file_sha256": "eb5ed8a5bcc4a0d830972f13f7a53db4169125805b099c9f3d16ba46a2bebfcb",
    "previous_transport_progress_file_sha256": "061635bd12af3fd206726d2c488ee673bee0d4ca4a8304b1798134ed40f2d11d",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795725838.6105614,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE18_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "33970472c3ba4a81963eaa903fcf1ef6",
        "job_sha256": "d2b3f2880dbf0ed91a547fa4841d6b34e08fac1bbca93e65645c67db7ccb2a9a",
        "result_sha256": "f7743c112bfa58179baf69a1b8ca7491638652c304efa10820f5ff4a5f05764a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "a4ca21791ef94e748d7f414a4b36ee6a",
        "job_sha256": "296e43ec72f4fe9268d94467fd0811f8fda2894cf31978cb1aa0b1cbcf8286c1",
        "result_sha256": "560db268fa4923f480008eceff8be087964aa316085b5aa5290195a03a7a638c",
        "error": "timeout",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE18_TERMINALS = (
    "b9682c0ac55fe8ef819b8ccf415575404dd3f961468849cca35335f6fd0b45a7",
    "eb5ed8a5bcc4a0d830972f13f7a53db4169125805b099c9f3d16ba46a2bebfcb",
)
RELAY_2357_WAVE18_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "9a63e5b1325420d18b1cd24a048c800e93582cdc5951cb7b22d3afcd4dc669c5",
    "attempt_00.replay_authorization.json": "89780823bf9c7ea7f8a41de1132a53322a887aa0653c352b06de9517fd1cf1ea",
    "attempt_01.replay_link.json": "ecde4dd5cb8ac497e0ef115cbdcfb9c22f6e7c54b6f29249239be9deeb008efb",
}
RELAY_2357_WAVE19_OBSERVATION = {
    "phase": "transport_failure",
    "error_type": "ReadTimeout",
    "event_type": "ipv4_preflight_http_401",
    "elapsed_seconds": 1804.6336837920826,
    "network_phases": {"direct_ipv4_preflight": {"duration_seconds": 0.8712731250561774,
                           "observed_seconds": 0.8765098340809345,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 0.8712731250561774,
                             "observed_seconds": 0.8744371249340475,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.4168070419691503,
                                     "observed_seconds": 0.420157833956182,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.22410049987956882,
                                 "observed_seconds": 0.6463227500207722,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 1800.004173916066,
                     "observed_seconds": 1804.6267460000236,
                     "state": "failed"},
 "send_body": {"duration_seconds": 0.807349625043571,
               "observed_seconds": 4.620853041997179,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.0001915409229695797,
                  "observed_seconds": 3.8131987499073148,
                  "state": "complete"}},
    "first_phase_seconds": {"http_phase": 0.420157833956182,
 "request_dispatched": 3.810599209042266,
 "transport_failure": 1804.6336837920826},
}

RELAY_2357_WAVE20_CONTRACT = "d84128e0f85eb584c4defd92003499f3b9a33f482c6d409548f363fd0cf3204c"
RELAY_2357_WAVE20_AUTH = "ec1fe4c70edcd97b28ae8118e0d43cffb0dd504b7eb3699b08de629139079329"
RELAY_2357_WAVE20_PROOF = {
    **RELAY_2357_WAVE19_PROOF,
    "previous_request_id": "76bc9f2e4fb0bf76780b7b5cde86008f56d051a02d2f15cb96956db667901d6a",
    "previous_terminal_file_sha256": "41ffc8a8d97689087fe32b7daf0b4f8e879f343b19863c65a32c1f58dcc54bf1",
    "previous_transport_progress_file_sha256": "476f22b37894c96ec270f9551c09a36e39084ec7308287ca4132089426da1f64",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795727020.8317,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE19_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "6babdfd158a0464fa5ed42e9a2fe50bd",
        "job_sha256": "92dd5d2e6aaf3986abd2bc2102c4388c0a08c4becb191043efe240992a8a1ad2",
        "result_sha256": "0401a0f57d7b505fd5a2494edb0f6e516f8cce5375086ac074401b85c0e63321",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "b7cb05a627f84af6bcf8315e3fc0a2c7",
        "job_sha256": "2cf1546c0c98ec26fb2659caab97ecf2d206e67ee2280b4c3435ba0acf0634bb",
        "result_sha256": "be2f33513a0d8c2ee018168aacac302028c6449131d96ec81e1b08bf747d4867",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE19_TERMINALS = (
    "192af9f7a98e0ea3f758921145b97f73c84810a896f2b130bb6aa438a2ab1c63",
    "41ffc8a8d97689087fe32b7daf0b4f8e879f343b19863c65a32c1f58dcc54bf1",
)
RELAY_2357_WAVE19_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "05db658e51edfc666fa5b9fda2aa5638a1c554258cd51e712719bb1b8d5a4654",
    "attempt_00.replay_authorization.json": "7e06679c9b8134c9c2fbb7f867356c23b20dffd9410091856e926e00cda24068",
    "attempt_01.replay_link.json": "ac1e4973b9abf9d960d3489d3008c99ccbe8bf3b23ce647a99587a912b86f446",
}
RELAY_2357_WAVE20_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "405c046b-b04d-4c0a-ba57-6ee85ae7e3d9",
    "response_id": "resp_0f06b0c9194e9ee9016b089d20b3f887d1bd8ecb7647ea8473",
    "conflicting_response_id": "resp_405c046bb04d4c0aba576ee85ae7e3d9",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 203.49816245795228,
    "last_sse_activity_seconds": 203.49816245795228,
    "elapsed_seconds": 211.15037337481044,
    "sse_milestones": {"response.created": {"elapsed_seconds": 31.156765665858984,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 31.158781749894843,
                          "interval_seconds": 0.002016084035858512},
 "response.output_item.added": {"elapsed_seconds": 31.161257416009903,
                                "interval_seconds": 0.0024756661150604486},
 "response.output_item.done": {"elapsed_seconds": 46.374838665826246,
                               "interval_seconds": 15.213581249816343}},
    "network_phases": {"direct_ipv4_http11": {"duration_seconds": 31.149189125047997,
                        "observed_seconds": 31.152318165870383,
                        "state": "complete"},
 "direct_ipv4_preflight": {"duration_seconds": 1.427600583061576,
                           "observed_seconds": 1.4329195828177035,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 1.427600583061576,
                             "observed_seconds": 1.4308044998906553,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.4483224591240287,
                                     "observed_seconds": 0.4517007078975439,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.23820479190908372,
                                 "observed_seconds": 0.6919662079308182,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 25.120877207955346,
                     "observed_seconds": 31.146710332948714,
                     "state": "complete"},
 "send_body": {"duration_seconds": 1.0123755419626832,
               "observed_seconds": 6.023157499963418,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.00016820896416902542,
                  "observed_seconds": 5.010503249941394,
                  "state": "complete"}},
    "first_phase_seconds": {"http_headers": 31.15373837482184,
 "http_phase": 0.4517007078975439,
 "request_dispatched": 5.008145582862198,
 "sse_event": 31.156765665858984,
 "transport_failure": 211.15037337481044},
}

RELAY_2357_WAVE21_CONTRACT = "2de8b2c7ea584ca8fa62262906212ea44eba8e8fb3f447b8e1fae7d12528fc98"
RELAY_2357_WAVE21_AUTH = "b71eb34048b44fc35ba4b44cdb72e76eed898e8b19ce67aa1988f2d48695e7b5"
RELAY_2357_WAVE21_PROOF = {
    **RELAY_2357_WAVE20_PROOF,
    "previous_request_id": "b6221350e82464e0f7cfcc8f4a1e3055b63d5c40fda361f075ffc970c6cbb150",
    "previous_terminal_file_sha256": "ba5175cd5179686b82d0e387ed9ad5a902588f28c31d6c87683e13e2b72b6806",
    "previous_transport_progress_file_sha256": "efb7fbb10c618c6ef56e5d98a523290cf3c3d92fcb6e8187c4a23cb72c28b071",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795728051.4221308,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE20_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "d3d52223789440a9bfa650ce7df19d34",
        "job_sha256": "47d066f6328e78705f3e7e6e1be53de19c26829301fcc3d8c2410a1f4939daba",
        "result_sha256": "f616149eef125ab0a90943a637cbe77ec386f74be2863ba9a32a4e853908b30a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "5732ea68547d4b0798f601e648609dee",
        "job_sha256": "3f7052fcba8ac14eb3ee2617a57306a3749c7d0054f8bb313654b9396e43b0fc",
        "result_sha256": "a20d3798a77d57e456e58f47d73894fc8216a0c4d1d5d3f4f5bccc08abed61d8",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE20_TERMINALS = (
    "574059ad3a5a86f12d6b906bf9d80466db808f95fc80e44dadc7052995ca8b63",
    "ba5175cd5179686b82d0e387ed9ad5a902588f28c31d6c87683e13e2b72b6806",
)
RELAY_2357_WAVE20_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "3a540a024b85272599c9f594aa0e98d2e157ae5418f330b9625b7fdd4fc04bbf",
    "attempt_00.replay_authorization.json": "ab35df7db5f4759938f49eecf09e0e457a22ff221cc150f9721f7d5880acb09a",
    "attempt_01.replay_link.json": "251ed732388bd0b0aa19e10e0d3333fe604da368fe8aef812364fa63a74618a5",
}
RELAY_2357_WAVE21_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "158e8426-960f-46a5-a130-2e8888a52cc3",
    "response_id": "resp_0afcc4244e6426d7016b08a131d4a087d29df4d0ef07a6d6db",
    "conflicting_response_id": "resp_158e8426960f46a5a1302e8888a52cc3",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 194.24631462502293,
    "last_sse_activity_seconds": 194.24631462502293,
    "elapsed_seconds": 196.9795132910367,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.0129694158677, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.01522958301939,
                          "interval_seconds": 0.0022601671516895294},
 "response.output_item.added": {"elapsed_seconds": 17.017461540875956,
                                "interval_seconds": 0.002231957856565714},
 "response.output_item.done": {"elapsed_seconds": 26.367308540968224,
                               "interval_seconds": 9.349847000092268}},
    "network_phases": {"direct_ipv4_http11": {"duration_seconds": 17.00531958299689,
                        "observed_seconds": 17.008419207995757,
                        "state": "complete"},
 "direct_ipv4_preflight": {"duration_seconds": 0.7112214169465005,
                           "observed_seconds": 0.7159068749751896,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 0.7112214169465005,
                             "observed_seconds": 0.7143886659760028,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.23093754099681973,
                                     "observed_seconds": 0.23430004087276757,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.2309152497909963,
                                 "observed_seconds": 0.4672612079884857,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 12.65418216586113,
                     "observed_seconds": 17.002798915840685,
                     "state": "complete"},
 "send_body": {"duration_seconds": 0.8904972081072628,
               "observed_seconds": 4.34403554094024,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.00018158298917114735,
                  "observed_seconds": 3.453265040880069,
                  "state": "complete"}},
    "first_phase_seconds": {"http_headers": 17.009910540888086,
 "http_phase": 0.23430004087276757,
 "request_dispatched": 3.450836875010282,
 "sse_event": 17.0129694158677,
 "transport_failure": 196.9795132910367},
}

RELAY_2357_WAVE22_CONTRACT = "2f5e7a5f9af4a903b1401d4f061b51b9a6fd9dfad2c0474754a0134d4648c035"
RELAY_2357_WAVE22_AUTH = "bb76dddf00fd94f1564913be2e5f3ab0571045ecba1b24128a0320b2e233b3ea"
RELAY_2357_WAVE22_PROOF = {
    **RELAY_2357_WAVE21_PROOF,
    "previous_request_id": "eb51a587aff6f4e7a4761ea218ed68a5163e30dfc140311fc47fa07226f979db",
    "previous_terminal_file_sha256": "ff4ad99d1554d7c7cbf9392cb99f630b2fdfa0e7411ee5cbd09a5c8542931bc1",
    "previous_transport_progress_file_sha256": "ff0d25471d74df99eef924cb392c7fed6aefc7048d13ec69f7ef915f71beddde",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795728993.6545773,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE21_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "61ca8f0d3b024d18a30097d130d3dd52",
        "job_sha256": "9d6722586a15fe95202e0e28469901cc0c5c0f9a29b59d97940139a2c13889fe",
        "result_sha256": "33d79ff4943d12c2df836c22e89e7c4357066465062021a1a228aa308e036208",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "8c97ccc80ce94f7cba136bfc8af99824",
        "job_sha256": "2887a6a6ecb9101d65770ee8abf0de4da9167b3b90afb6105667d13de873edaf",
        "result_sha256": "3eabbf13228e33bde8eb4deb1ed0aef9f5fcd9b85a87d8c0b18898454355f34a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE21_TERMINALS = (
    "29bd0e510d1b94e7a444b3438f19e3b687ca21d2705481ee887dfa9249c0b43e",
    "ff4ad99d1554d7c7cbf9392cb99f630b2fdfa0e7411ee5cbd09a5c8542931bc1",
)
RELAY_2357_WAVE21_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "595698cead04d8cc320ab403e6c05494150461b64d5002b26a4e14b828946ed6",
    "attempt_00.replay_authorization.json": "8957527ada6db3bf72afb1dff7cc16832c50b04a924d530bfbf449f6fad0a558",
    "attempt_01.replay_link.json": "c2cebcf186572b0cb5bda9ccb60e29a6c5ee0f1cfa60c3f7a2594d41cd30387e",
}
RELAY_2357_WAVE22_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "a4f2e440-a571-42d9-9528-f2e89d5d27d8",
    "response_id": "resp_0f68cf21289a73f2016b08a4d3492487d0983c58a9d4449ff9",
    "conflicting_response_id": "resp_a4f2e440a57142d99528f2e89d5d27d8",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 200.32678483310156,
    "last_sse_activity_seconds": 200.32678483310156,
    "elapsed_seconds": 207.71423870814033,
    "sse_milestones": {"response.created": {"elapsed_seconds": 27.688676041085273,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 27.691013916162774,
                          "interval_seconds": 0.0023378750775009394},
 "response.output_item.added": {"elapsed_seconds": 27.693921083118767,
                                "interval_seconds": 0.0029071669559925795},
 "response.output_item.done": {"elapsed_seconds": 41.860319041181356,
                               "interval_seconds": 14.16639795806259}},
    "network_phases": {"direct_ipv4_http11": {"duration_seconds": 27.680657875025645,
                        "observed_seconds": 27.684117125114426,
                        "state": "complete"},
 "direct_ipv4_preflight": {"duration_seconds": 0.758600499946624,
                           "observed_seconds": 0.764687665970996,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 0.758600499946624,
                             "observed_seconds": 0.7622066251933575,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.1926773339509964,
                                     "observed_seconds": 0.1965194579679519,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.1985441658180207,
                                 "observed_seconds": 0.39667525002732873,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 23.544121667044237,
                     "observed_seconds": 27.67908920813352,
                     "state": "complete"},
 "send_body": {"duration_seconds": 0.7592190410941839,
               "observed_seconds": 4.1326604161877185,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.00014233309775590897,
                  "observed_seconds": 3.3731649580877274,
                  "state": "complete"}},
    "first_phase_seconds": {"http_headers": 27.68557966616936,
 "http_phase": 0.1965194579679519,
 "request_dispatched": 3.37110049999319,
 "sse_event": 27.688676041085273,
 "transport_failure": 207.71423870814033},
}

RELAY_2357_WAVE23_CONTRACT = "1d4544da7e7f97d4a1073b95602dd8cc0552092d8a0357d74e0793440ec2a9b5"
RELAY_2357_WAVE23_AUTH = "8f58632ec77f732357a0b7e4e9f40b5a6b068ae4aeea830a030ffa37e97fdde7"
RELAY_2357_WAVE23_PROOF = {
    **RELAY_2357_WAVE22_PROOF,
    "previous_request_id": "53fb823c8dd341349d631495e056b5740f72fa7caa05e16bb424bf0221091ea2",
    "previous_terminal_file_sha256": "952bc63f4deacdfaad9f7f310ed2c593020ea10def642325ce7477c7100ed486",
    "previous_transport_progress_file_sha256": "ff87e785259deda019b25d2aac70bbd26ca31e4e0609f60914a652474e502036",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795729955.532721,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE22_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "7497055372d545b4baeb0c3ee3d6705d",
        "job_sha256": "dc8b1a4270af653c68717b6bdb4a07558d1f8759c6e42a39096c2d38bf7a5331",
        "result_sha256": "aff09024a572ab3684742d6fe14cd0565b5a79be44c9a46c3fd03f6cc817e0ca",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "165867cccd6b46678fb7d6238adfb53e",
        "job_sha256": "b1b55820e277d79883c3bec7ae72d2496f9e8585db384ef65c572d958dbdb6db",
        "result_sha256": "48b248ec4e759a6ca797e3369b7bdc6d8861a4f872d7bc50db4720666328a2da",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE22_TERMINALS = (
    "e3a394aa2503e3510b084d9c0a7a0135d92924f3fb0de92ebd929c10dc62b777",
    "952bc63f4deacdfaad9f7f310ed2c593020ea10def642325ce7477c7100ed486",
)
RELAY_2357_WAVE22_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "14d1db3b694faf7bf65e3d0a112524ea8933fc543d01f2f8df1d92ed6765e3a8",
    "attempt_00.replay_authorization.json": "1ba6833d2bcf1f72bf179eafb686ccfecec3e870c4d760402af3a44c5c070252",
    "attempt_01.replay_link.json": "8e7bf861d31f52666f857cc0ede7de7765d24484449d81d9837a343053951648",
}
RELAY_2357_WAVE23_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "d3397732-79e5-45fd-97db-e297be7c7a65",
    "response_id": "resp_057dfee8e10ebbec016b08a895091087d09081252127029931",
    "conflicting_response_id": "resp_d339773279e545fd97dbe297be7c7a65",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 207.89976579113863,
    "last_sse_activity_seconds": 207.89976579113863,
    "elapsed_seconds": 210.13794500008225,
    "sse_milestones": {"response.created": {"elapsed_seconds": 30.186397791141644,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 30.18774054106325,
                          "interval_seconds": 0.001342749921604991},
 "response.output_item.added": {"elapsed_seconds": 30.189011791022494,
                                "interval_seconds": 0.0012712499592453241},
 "response.output_item.done": {"elapsed_seconds": 46.978341083042324,
                               "interval_seconds": 16.78932929201983}},
}

RELAY_2357_WAVE24_CONTRACT = "eff07edbf67909521ca4214587fd849d0364a383be5b63c227fbd442b62474ab"
RELAY_2357_WAVE24_AUTH = "5e952d6748929285770425bdcd9e1117585ed78719a69f54f8c0ac3a26585018"
RELAY_2357_WAVE24_PROOF = {
    **RELAY_2357_WAVE23_PROOF,
    "previous_request_id": "16907a55840c66c655bc7a166bd050d24d463a9e9356109d86ea412c8ce872d8",
    "previous_terminal_file_sha256": "cbb859d09251fcaacff360a360b59f5b4ae25224cdf26b3a97245ba550dfbcac",
    "previous_transport_progress_file_sha256": "9fc52d20cb52972a7c9aadaae7c6ebeaadabb30f892cdc4ac5fdc54796e3fd51",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795730986.2878697,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE23_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "c08afd5215ad43f7bea313fd70b0b32c",
        "job_sha256": "3c043793280766350efc5c7070977f7e09b5abd9e4fd99de45eec1e00c1f5385",
        "result_sha256": "c6f16be59f1934086749edf118ee3a03f122daaf5b3c9b4619d03de930d473f9",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "188beb8c58bb4483b724f55ae7b8a4a7",
        "job_sha256": "1392f69bbdd26e22a0e5f5e0efeca59084fb87203fc8e5148fe1873bfec47ad9",
        "result_sha256": "455a9ec44e4758ce514659790955dd60bdc8181233fae3d33f3180167075e6a1",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE23_TERMINALS = (
    "e8cc17354c9aa2ee79a3ad97e62562718f770815388dbc0e77d99d0a619803a9",
    "cbb859d09251fcaacff360a360b59f5b4ae25224cdf26b3a97245ba550dfbcac",
)
RELAY_2357_WAVE23_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "ed21087d2d4608b8bac955f78179c1ff9f3bcdae4f9220f5aab62469856b7e90",
    "attempt_00.replay_authorization.json": "1cae94b20aaba7d83ab2d86828dcd6475f0a68a6f4f2923104a09474d47a0aaf",
    "attempt_01.replay_link.json": "e10421c1b8218c81c948cc8a64ab812fc3c8f1b1e349abc726fb4e2a2ca95605",
}
RELAY_2357_WAVE24_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "32f9a784-2f6c-41e7-b686-66d074cbb300",
    "response_id": "resp_0f4dce7737962e7b016b08aca6fd5c87d2ae703cae6675280a",
    "conflicting_response_id": "resp_32f9a7842f6c41e7b68666d074cbb300",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.03721862495877,
    "last_sse_activity_seconds": 195.03721862495877,
    "elapsed_seconds": 197.8289520831313,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.904734125128016,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.906998415943235,
                          "interval_seconds": 0.002264290815219283},
 "response.output_item.added": {"elapsed_seconds": 17.90989683312364,
                                "interval_seconds": 0.002898417180404067},
 "response.output_item.done": {"elapsed_seconds": 27.42339575011283,
                               "interval_seconds": 9.513498916989192}},
}

RELAY_2357_WAVE25_CONTRACT = "21aa2c0242cf18232fcdf5d84291a4297d66ab73007cb096dd7637c86b81cd61"
RELAY_2357_WAVE25_AUTH = "3c9a1e2d99ea749830f31507b9cd7dc4e7665ba2b7d69a725ceebc6339094ead"
RELAY_2357_WAVE25_PROOF = {
    **RELAY_2357_WAVE24_PROOF,
    "previous_request_id": "bb9f8f141044c9eaa818fe45072d2e2d694a7e3d6c14b68a154067fc6b2de998",
    "previous_terminal_file_sha256": "3caa6ad484a5a891f317add9738133d6938f9e8a32e5d3c775f020e06a653108",
    "previous_transport_progress_file_sha256": "3d8e6d77b11fcb25f51efe5bc22950284af3704bb33d16d692647c991cdca36c",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795732017.8086386,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE24_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "fd16a4a08e484174979bb12299314010",
        "job_sha256": "944bb844b48db27130483a39e6f88f840aea4fb3db15cac18d86d55013b69966",
        "result_sha256": "5676af416b8e64712624af5d16bb9c196be302d44a8cee52f694a2cf90037048",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "c938c9841320450b888100333327d8ee",
        "job_sha256": "8f634a8c5ce685b42cc8798a58c7fc332fe581a79f188ec2a2d1f2e9d318ef04",
        "result_sha256": "26ae65763bf546d474bba13c4ee56c67eec1269af37883ece8f6740a954e3ee9",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE24_TERMINALS = (
    "b82f75d3e6c1368c049acb4e183efb7e5da5f2ab18a775976b62fbfddc47c4b9",
    "3caa6ad484a5a891f317add9738133d6938f9e8a32e5d3c775f020e06a653108",
)
RELAY_2357_WAVE24_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "2b68b0464717cbfbba598715784fbd44dcdeb124c19eb12035f9fb6c1609e606",
    "attempt_00.replay_authorization.json": "fdb3dbbe918ae254736a9477b3a6974b5a09b9895250e846fcc1614c5996dd94",
    "attempt_01.replay_link.json": "b23e9cb9e4621a26180885293f1e1b0c8d6299e2f60bd16facf772684c88c62a",
}
RELAY_2357_WAVE25_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "581d527d-4e08-4faf-a014-effaf339f5c1",
    "response_id": "resp_04d26373edb3d28f016b08b0b3889487d2b04818b0cf76561a",
    "conflicting_response_id": "resp_581d527d4e084fafa014effaf339f5c1",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 194.02827716711909,
    "last_sse_activity_seconds": 194.02827716711909,
    "elapsed_seconds": 196.79359487514012,
    "sse_milestones": {"response.created": {"elapsed_seconds": 16.841802709037438,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 16.843677750090137,
                          "interval_seconds": 0.001875041052699089},
 "response.output_item.added": {"elapsed_seconds": 16.84644133411348,
                                "interval_seconds": 0.0027635840233415365},
 "response.output_item.done": {"elapsed_seconds": 26.225268041947857,
                               "interval_seconds": 9.378826707834378}},
}

RELAY_2357_WAVE26_CONTRACT = "7914a06ecff6d4a30fc9ce789188e9cc8758a2cf34415cb33b2c0af453e2b443"
RELAY_2357_WAVE26_AUTH = "39835e467d061e2eade40505e4ea264dbb65e086717ce2665e65c639cb7da6f4"
RELAY_2357_WAVE26_PROOF = {
    **RELAY_2357_WAVE25_PROOF,
    "previous_request_id": "a827fd2164115b60ade9a3c1054b8fa63067dfe6d5f93e3f3e02ed801319baf7",
    "previous_terminal_file_sha256": "4faaf9566190c9de4c1dad47ec76198d64c104a4a07898fd9c153c3d05a564d9",
    "previous_transport_progress_file_sha256": "c98b680a701ec706ea1cc3bd666e339bf628416e77271efa0392c59d75106bbd",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795733165.5820577,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE25_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "54e6018e7f3f4a69a085dfc19068f24a",
        "job_sha256": "7d3ec52e4574005f716f25925a167ad89a02fca44055a2fa259e8f17308e5679",
        "result_sha256": "47831745ff0e225e8d1a880615245ec0c6c8f8e210a96e1e44ab905e58d5bf9e",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "097a914930474894a4fe5ca1de033dc1",
        "job_sha256": "2f06fb7d1ac36df78b04e39e2f8299d619fe0ba8fc7aa1b75df5016a78a7d0ea",
        "result_sha256": "f76067fa1b69e6384c12a8f8b3879b6e4683bd57438236dbe14d0f16608f31c9",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE25_TERMINALS = (
    "3484e73db6357940fc2f7c2bb7d8346021edb9bcf5cc5c46404d8c311c4d720c",
    "4faaf9566190c9de4c1dad47ec76198d64c104a4a07898fd9c153c3d05a564d9",
)
RELAY_2357_WAVE25_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "204eb5f625e4c87301bbf6d8c819e2b32c8c5fb562113b96f071788dfaf05f73",
    "attempt_00.replay_authorization.json": "2833fe49c477f5e1cb70ffeccb6aefb48f898fcf540502d86f45742b321a0e6c",
    "attempt_01.replay_link.json": "48fd7e90ae46be47bb04486305a3deb5c8e7a604b277850431c0f794debae8a4",
}
RELAY_2357_WAVE26_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "12a86783-4e21-4e79-bd11-4b7f3d572de4",
    "response_id": "resp_0af1db14561c9ec7016b08b52c881887d29fafe189d5ae0de5",
    "conflicting_response_id": "resp_12a867834e214e79bd114b7f3d572de4",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.06362916599028,
    "last_sse_activity_seconds": 196.06362916599028,
    "elapsed_seconds": 198.72531612496823,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.793904041172937,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.795538750011474,
                          "interval_seconds": 0.0016347088385373354},
 "response.output_item.added": {"elapsed_seconds": 18.798236125148833,
                                "interval_seconds": 0.002697375137358904},
 "response.output_item.done": {"elapsed_seconds": 28.116756916046143,
                               "interval_seconds": 9.31852079089731}},
}

RELAY_2357_WAVE27_CONTRACT = "4cd0e5a07e221cd09f9355b0400f86f28eb000ac5504c0e979d4da6830c664d5"
RELAY_2357_WAVE27_AUTH = "77c1b9377bf82fa49d113484f7ee9e855ba65b137d08103778a498b31f6c02ee"
RELAY_2357_WAVE27_PROOF = {
    **RELAY_2357_WAVE26_PROOF,
    "previous_request_id": "3388c48537c7aed58993611fae485a21d20afe0777ea6bdc094ae4bbbd796e83",
    "previous_terminal_file_sha256": "9755f371d1e4e2c6144e129cc4f94b93edd93c8e431b5d8dfc233694401a2ac8",
    "previous_transport_progress_file_sha256": "f6d60b85763355643ea686a55b93753e5b9f73f7eb926395b8d7f5ef39ea0d8b",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795734273.7636313,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE26_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "0ccb81fe134c44f0bddd832be74b011a",
        "job_sha256": "5d73e94c0457af8e0eaa89bb2796dc56cf47447645b652ff0da4ca6264dc037f",
        "result_sha256": "d6a92f52f4c8753d70d6c9b37e18704f9a65005d62067c70e070120be686714b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "9be434ac07fb46c695498bce09632916",
        "job_sha256": "6d800a3138288f3bd981d3380fbb49d6c5b9667a943aed2102383b458bcc9a79",
        "result_sha256": "f68e705121ce6f27a67bc771ec2e127de3b855afd636b190f5e9072d1cb2f3fe",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE26_TERMINALS = (
    "9aea8facae60767aca52e842147a17a5f3df3afc8103850edfadb2e4413a2591",
    "9755f371d1e4e2c6144e129cc4f94b93edd93c8e431b5d8dfc233694401a2ac8",
)
RELAY_2357_WAVE26_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "4cda18d6fcc22368189aa30087d425659aa846ae219fbabe5f93a9663101bcec",
    "attempt_00.replay_authorization.json": "f31dbc10fefbe04dd206e5c4019a80bd70722f25647b4aca964753e98ed930f7",
    "attempt_01.replay_link.json": "0fa90bf0207136374e16525a836c62331325bd3db835897aa4a59e31aaaff889",
}
RELAY_2357_WAVE27_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "4d03255c-4500-4be5-b037-e5a4994df185",
    "response_id": "resp_036cb490d4e7c5bf016b08b97d33d887d2bb9ecfe550ece8fe",
    "conflicting_response_id": "resp_4d03255c45004be5b037e5a4994df185",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.2564327090513,
    "last_sse_activity_seconds": 197.2564327090513,
    "elapsed_seconds": 199.87424100004137,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.93588316696696, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.938056374900043,
                          "interval_seconds": 0.0021732079330831766},
 "response.output_item.added": {"elapsed_seconds": 19.94104020902887,
                                "interval_seconds": 0.0029838341288268566},
 "response.output_item.done": {"elapsed_seconds": 29.267843541922048,
                               "interval_seconds": 9.326803332893178}},
}

RELAY_2357_WAVE28_CONTRACT = "10e8f14f18c17901c8bd512e0f8b8de0b32c1ceb6436111f656c55593eaabd7d"
RELAY_2357_WAVE28_AUTH = "4893da428bd45bfb000b090649f406bad9aac968836fe0f87c10d0724b49d62f"
RELAY_2357_WAVE28_PROOF = {
    **RELAY_2357_WAVE27_PROOF,
    "previous_request_id": "9e4c963e12afe64018292538eb7894439f54cdf43d0363bc0f244f745926d5c6",
    "previous_terminal_file_sha256": "d1b95c66936cca86341faafa872b278aab4c6b14aaaa7c11ec47e3efbf063616",
    "previous_transport_progress_file_sha256": "b4fa614471031c7182c0fc11c8d6ad7a98b348ee54bf90b612374e6d643635a3",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795735310.6910114,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE27_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "b5becb27b73d441abd542893c7ebe1fb",
        "job_sha256": "27ebf802ab284e0beb479dca12e3889f92634e68370a974c7f355172b5b96809",
        "result_sha256": "e4c32a8fa173d4403351907f82b2d6ea5ea68f8babef097e451684c83f05107d",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "22c3be0ae6a043ae8767b47fc0cc4f10",
        "job_sha256": "d8758355a33ab8dffd10c8a0579531a123af1c8b02b28070dee6b7f09df6fc3a",
        "result_sha256": "010ea38bbfbe17105ebf4c318b603feb25512e48f17f3962eeedc3e254779a1b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE27_TERMINALS = (
    "4d2d3fceae85f1590b082031a1f96de5bc06238615881563af8e39e5a659325d",
    "d1b95c66936cca86341faafa872b278aab4c6b14aaaa7c11ec47e3efbf063616",
)
RELAY_2357_WAVE27_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "f62c1ea38a47a8a766f4968d8363ca6c4396eb4cdb7876e6daee2d35c905e79a",
    "attempt_00.replay_authorization.json": "e9e47554e2db139737d648a519fbc4150315e574d840b44eef212dfb1290980a",
    "attempt_01.replay_link.json": "1dce663552e51a7fc50bf486a85f7ec7de18e2391744e881fcc735f4f1d2e780",
}
RELAY_2357_WAVE28_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "6afbd19b-1bac-45a5-8c9c-6f48336da74f",
    "response_id": "resp_09690a1667ba5121016b08bd8af28c87d1a4197aa26a3d6e2c",
    "conflicting_response_id": "resp_6afbd19b1bac45a58c9c6f48336da74f",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.53409325005487,
    "last_sse_activity_seconds": 197.53409325005487,
    "elapsed_seconds": 200.19157704198733,
    "sse_milestones": {"response.created": {"elapsed_seconds": 20.255071375053376,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 20.25707745901309,
                          "interval_seconds": 0.0020060839597135782},
 "response.output_item.added": {"elapsed_seconds": 20.25938383396715,
                                "interval_seconds": 0.00230637495405972},
 "response.output_item.done": {"elapsed_seconds": 29.500921583967283,
                               "interval_seconds": 9.241537750000134}},
}

RELAY_2357_WAVE29_CONTRACT = "9396209a943b7dd2e56de20959ffe894d6ebad06edfaeb6f15f66258d0b77f29"
RELAY_2357_WAVE29_AUTH = "0ffa3fed02e934abc7e8784861e0d6f74c4e50f5aa07c754c97712a62afbf60a"
RELAY_2357_WAVE29_PROOF = {
    **RELAY_2357_WAVE28_PROOF,
    "previous_request_id": "2a05821649ce44291156d12ec60ccd2f8d9763bbf2283dab1b7a103f886db215",
    "previous_terminal_file_sha256": "c8d52e745c6b9ec40ba06f47392e9b545aeca9f61fd41f6199e52f7e51c462cf",
    "previous_transport_progress_file_sha256": "538ba8f60c6b6a107651b8bd881aca58cdd2b97d3aaa1599acb428da03e7889e",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795736317.8906271,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE28_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "54682c14ff5a48dd904957b6b36272be",
        "job_sha256": "4f017118b896d6f86bbbc706bb537949d460ee5d36f3e6f3b675e992b7130613",
        "result_sha256": "0eef0a72dd867e6f312627816d21891a847e474529e25c1cc43275f8001c69df",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "a23d5a96ea9942d983bdb76b11234c9e",
        "job_sha256": "e70281e69d2d32c6eb6ba5be1a7e384e8126899f62571050ad79848d77fb6ac0",
        "result_sha256": "901c80bc1ca4300d1e0fe415fe09de6a9651246e31cf0d7d364a4aadc1dcf674",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE28_TERMINALS = (
    "5e1dbae6dd15933e0ec02f7d7ea37bc305281e62b0213f1243ed3e92c81edafe",
    "c8d52e745c6b9ec40ba06f47392e9b545aeca9f61fd41f6199e52f7e51c462cf",
)
RELAY_2357_WAVE28_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "81bed570f2bd1fcffc172377f58fc0a42c2f80470cc7f9a1dc86b5cfcd00387e",
    "attempt_00.replay_authorization.json": "a2bc2bc31557164cf00e3c48e38d1d4653b2368a37b3e9cb8dd2b9163a370539",
    "attempt_01.replay_link.json": "6f5e69b57415d5a1ee88877b634aeae5439196225d219e285d2bfb6d8fccc163",
}
RELAY_2357_WAVE29_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "fcad5f9f-4c6a-4287-a2f8-594310718f9a",
    "response_id": "resp_0819253a1f5835a5016b08c179ceb487d0afe64f067a3f87b1",
    "conflicting_response_id": "resp_fcad5f9f4c6a4287a2f8594310718f9a",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.72695729113184,
    "last_sse_activity_seconds": 196.72695729113184,
    "elapsed_seconds": 199.35806600004435,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.354329250054434,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.356844207970425,
                          "interval_seconds": 0.0025149579159915447},
 "response.output_item.added": {"elapsed_seconds": 19.358802624978125,
                                "interval_seconds": 0.001958417007699609},
 "response.output_item.done": {"elapsed_seconds": 28.683909208048135,
                               "interval_seconds": 9.32510658307001}},
}

RELAY_2357_WAVE30_CONTRACT = "dc8bb2ab250071bb8680b09c413e67fac577dee5dfebe97f42655be43367d01f"
RELAY_2357_WAVE30_AUTH = "ea9a9e0a3c700a9a7df378cdef47de23baba34d65750aa6d8728108286f26385"
RELAY_2357_WAVE30_PROOF = {
    **RELAY_2357_WAVE29_PROOF,
    "previous_request_id": "7afbb80edb4f8863d3dad46a8479d232ea9de0ed75a253d67b8f81374f838400",
    "previous_terminal_file_sha256": "76a2c187a3c2a854ea6ece2cc37dec4af206344f2b3a9d4d06b218e318f6986f",
    "previous_transport_progress_file_sha256": "1607ce9d93c52e0eeeabb7514aa3bbf319bcbe792292a5ef352c37882a5c754a",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795737316.2258618,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE29_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "9269d180774d4e9490a49687a1745253",
        "job_sha256": "78fbbd9c94bc21a9bdeee3eb832a0cd4ede2d1b40a8eebe42ee0f6eb84744f15",
        "result_sha256": "b1206b8253f6a6c6a49fff75330a616da15b7bd8f123ab0a60796df9ac03ce1b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "01f17f4d387a4b55b8df14c7cc20faf2",
        "job_sha256": "0822135ed1b98c31c010508adfcef2fae4e671184051512392b17d276fb43942",
        "result_sha256": "45d058c904a46b035ac81966ab1b58769c7856408bff300a554bc6d019d9d48a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE29_TERMINALS = (
    "ef303e5b6a43d2e6fcf4eb751b82dcfeb3a2d4d23b98747e75d631e84d55889f",
    "76a2c187a3c2a854ea6ece2cc37dec4af206344f2b3a9d4d06b218e318f6986f",
)
RELAY_2357_WAVE29_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "3752614605112a901a3b0ad3b7f1062950339c27b18679ba782126282bd2dd7a",
    "attempt_00.replay_authorization.json": "ee1ae0f078a6952e374554165ed78f3a5bca80058f2bd0cb3b460f7378720ecc",
    "attempt_01.replay_link.json": "90ac339b0fa83d2c2f3e8f604d53b34fd9762c5ef382a953583d41fa0331f90a",
}
RELAY_2357_WAVE30_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "342f5099-dc4b-41ac-a256-3b05e62f9808",
    "response_id": "resp_0374df729b8b374f016b08c55df77487d1a7905b726b056287",
    "conflicting_response_id": "resp_342f5099dc4b41aca2563b05e62f9808",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.32238687505014,
    "last_sse_activity_seconds": 197.32238687505014,
    "elapsed_seconds": 200.0266090829391,
    "sse_milestones": {"response.created": {"elapsed_seconds": 20.084737458033487,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 20.087405791971833,
                          "interval_seconds": 0.002668333938345313},
 "response.output_item.added": {"elapsed_seconds": 20.09039287501946,
                                "interval_seconds": 0.0029870830476284027},
 "response.output_item.done": {"elapsed_seconds": 29.380561749916524,
                               "interval_seconds": 9.290168874897063}},
}

RELAY_2357_WAVE31_CONTRACT = "62c76f019af05ffbcef352171686fb1f77b47c07aad7bc17d9cf37e1f50b86a0"
RELAY_2357_WAVE31_AUTH = "0b776df21e0ab70d71bc8cc1b2ec666d7afa4a7c05ec7e0cb0503b8603ef2d2a"
RELAY_2357_WAVE31_PROOF = {
    **RELAY_2357_WAVE30_PROOF,
    "previous_request_id": "25c75a7d93015c1503310e948fc30f13456e8e6484a5c40c9f725425b175da92",
    "previous_terminal_file_sha256": "80eaebf17f56c1a52060d5964af2b0704a6f07fb9cd9b4fc8c14bb77a5ff9142",
    "previous_transport_progress_file_sha256": "d5245236e6953a107f2e76701c392e65614f7c3009b66776bd4063c7ac253b28",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795739518.5515773,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_2357_WAVE30_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "0f819345a2d54c579c14ae79cb126521",
        "job_sha256": "887fa0354f372782c89b05e04899ad9b7adf77f0547a2c7c43e5885500c89e18",
        "result_sha256": "759b9f2bb0c39a475b92cdb7afc09f981d177bb3eef37a2f122fabfacb1cb3e8",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "91de0b3336a94cfb989c583d5a53531b",
        "job_sha256": "9ec54259a3613bca742b4db1eb8b401f1e5b7bdedef4e495c000481efbdd30c0",
        "result_sha256": "4fc60a3a47be789fef329b99829c48cbd2234757e55d87bb573714c61770118a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_2357_WAVE30_TERMINALS = (
    "d5ba190b1f6a94d3e8c3ccf17866cd69478ff47da6611a4e1c42ef0d796ba515",
    "80eaebf17f56c1a52060d5964af2b0704a6f07fb9cd9b4fc8c14bb77a5ff9142",
)
RELAY_2357_WAVE30_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "3d18ccb10e28c606efce84123ef5f821ee5c44915022b30a259a15ef452d06e7",
    "attempt_00.replay_authorization.json": "e650c656d3538f7bbd2366b498ba7d85d6c89e30fb1302d3552d1e662c5636e4",
    "attempt_01.replay_link.json": "a9a3dc889b36cbe513a6d1282f84a0e55c9a296fb8a46d7cda8d3fb1ff65fef0",
}
RELAY_2357_WAVE31_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "287e6e20-4ccf-4af6-9c98-25064cbc4d5f",
    "response_id": "resp_0438c3370fbcdeca016b08cdfffea487d191d0d8369c6f5a14",
    "conflicting_response_id": "resp_287e6e204ccf4af69c9825064cbc4d5f",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.29405379085802,
    "last_sse_activity_seconds": 196.29405379085802,
    "elapsed_seconds": 198.92930995789357,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.963850582949817,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.96606179093942,
                          "interval_seconds": 0.002211207989603281},
 "response.output_item.added": {"elapsed_seconds": 18.970126874977723,
                                "interval_seconds": 0.004065084038302302},
 "response.output_item.done": {"elapsed_seconds": 28.33027408295311,
                               "interval_seconds": 9.360147207975388}},
}
RELAY_2357_WAVE31_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE30_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE29_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE28_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE27_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE26_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE25_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE24_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_2357_WAVE23_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_2357_WAVE22_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_2357_WAVE21_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE20_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_2357_WAVE19_ARCHIVED_SSE_COUNTS = (41, 0)
RELAY_2357_WAVE18_ARCHIVED_SSE_COUNTS = (41, 23)
RELAY_2357_WAVE17_ARCHIVED_SSE_COUNTS = (23, 41)
RELAY_2357_WAVE16_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_2357_WAVE15_ARCHIVED_SSE_COUNTS = (23, 25)
RELAY_2357_WAVE14_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_2357_WAVE13_ARCHIVED_SSE_COUNTS = (23, 23)
CACHED_R7_PLAN_SHA = "d8acfb0d43235068f932f18b7f09c08554e811060f04878fe1a5a9afe8386d47"
RELAY_887_CACHED_CONTRACT = "85f7448d3f2bf60e28b778be01cceca626fc9c26f03a5f32c1b6bf0fd2777763"
RELAY_887_CACHED_AUTH = "c87bd63366932852ea9a96c40346446fd8f5aa3091e68b851280a9bf67f62f9d"
RELAY_887_CACHED_REVIEW = {
    "proof_sha256": "20c979db30ef6f959e873df178d42cef2ab5b000d857e564b59e92b4d6b8d5b7",
    "receipt_file_sha256": "cf345efd407710e3a11d34196b5bd7befda0a229e552f1745d9848b1c5dccb28",
    "resource_contract_sha256": "f4e2782ba061c1c00dd9a1f46198380bc095492e38b467e6d9bf20b4b828b33d",
    "resource_authorization_sha256": "37bcb9bd06d31622ce29ff56363bf1d9bfb8772e2e3016e133fb3e9bcd29b7b3",
    "provider_post": False,
}
RELAY_887_RESOURCE_SHA = "c44a0d1a1f2961e99c5f1cd0032599259de09eef77a3aed93480cfb658ec5cf1"
RELAY_887_FITTED_PREFIX = {
    "round_08/source.json": "44b96e6448131a1a37eb3a0b2a80d8a13eaca7c7e16945ce85686f1b9eb039e2",
    "round_08/transition.json": "7ca5f9400e1a3152a0cc9e920bfdc6fb90739bf0a872d811b8d364bc39217c72",
    "round_07/memory.json": "771b3cbbe0f28005ff04a9dab30084035b9e1eebd70a3e5137dc9cee1d367de1",
    "round_07/batch_freeze.json": "1e0e41cb74a3525d7b30f3d80d1c6aae3e65d031eb8699b3f35c3caa404d37f4",
    "round_07/turn_00/attempt_00.terminal.json": "9324a781e9596a7ed994ad9e275aad7aca02cbfbfed04ddfafb6276430ef07dc",
}
RELAY_887_R8_ATTEMPT = {
    "attempt": 0, "attempt_id": "fa7a605ce7f0490fbe8a39854981dd38",
    "job_sha256": "4218b6a7f7dfbb4940aff87d879ed297578f09c77fa4775f65e13735e4b73ecc",
    "result_sha256": "6b77162673d7ead51444c80c400a3c5195e4777beaf333b5fff5c6d200a31fe4",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_887_R8_PROOF = {
    "logical_index": 0, "payload_sha256": "ac09bcae3019cc81899e246e4769bc20ec1b764ac08e52cfc6b4c35dc8324c85",
    "previous_request_id": "299309f8e73e0c13a720b0537a368c61ee691d9d109f828c49dcf5e24c8eea22",
    "previous_terminal_file_sha256": "09a9dd7c2ceb5aee0e6a3188c9ac9ce1cb6175debc530e015c8fed18b316f2c2",
    "previous_transport_progress_file_sha256": "e4b2958bfab5d6c0f75db71c545b88ee07899a72a5270a70c30d85f9a9d4d6ca",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795685708.1872327 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 163},
}
RELAY_887_R8_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "2b53b779-42cc-472e-8b2c-509604178690",
    "response_id": "resp_06e929eed61879f8016b07fc5dabdc87d2a1e2ec641e7fabc1",
    "conflicting_response_id": "resp_2b53b77942cc472e8b2c509604178690",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_887_R8_SOURCES = RELAY_RETRY_QUEUE_SOURCES
RELAY_887_R8_WAVE02_SOURCES = {
    **RELAY_887_R8_SOURCES,
    **{f"studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/{module}.py": sha for module, sha in (
        ("run_v13_retry_successor", "187f99aca94eda0a045c48f9086661bf600822cd5131f73ca6c1b549fff1cbbb"),
        ("dispatch_v13_retry_successor", "bab19494a7f79116589f2eeeb53dabc5687ab14df785688305601fb28a7073a5"),
        ("run_v13_relay_retry", "e1dc27f624fd2d4021ab1f225da235efcc676c71a31c11ba787ec1858a285ba5"),
        ("dispatch_v13_relay_retry", "ad33b243aaff6905a65396517ebf21e2060b6310ef0242028001594fe49109af"),
        ("run_v13_12203_relay_retry", "6413ef582b5f746a9b640021ae386d0fc4c1d40d1d615cb944bcd277340e864b"),
        ("run_v13_157_relay_retry", "9745c33a707dc49deba8b28dccde3635d463f0b8329225a1e63d992b84981460"),
    )},
}
RELAY_887_WAVE02_CONTRACT = "a0e19b519b4cd3c96038563fb470f93ab5b5ff9d943aee4215292a256ba50f3e"
RELAY_887_WAVE02_AUTH = "80c19a60bda6db98a5ec2f7b98051408b027b1dec7889b11f89a7288c3537252"
RELAY_887_WAVE03_PROOF = {
    **RELAY_887_R8_PROOF,
    "previous_request_id": "33dd5e0f75cc3e5b932126505a2a2aae1bfd76fbb9f77394b1d1cb94815a4520",
    "previous_terminal_file_sha256": "4d8908c38af866b174465006e0d65bcacd47a6d1ba85af658f0c76c9c038634d",
    "previous_transport_progress_file_sha256": "c3a801b2c92eca99e825e30f523e0025d7065fc775e8a73853a066b803ca9b92",
    "relay_not_before_unix": 1795687017.5816553 + RELAY_RETRY_SECONDS,
}
RELAY_887_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "844b413de8424d998374e86a31535b48",
    "job_sha256": "b12ee1189f1d36bd35f1eccbfd6a80caf8962908864f0ba7438feb092798372e",
    "result_sha256": "6b3284b78b60617b0f45befb1913d21f33ff20859122f2d75749f9a13be80719",
    "error": "error", "provider_dispatched": True, "ambiguous": True,
}
RELAY_887_WAVE03_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "event_type": "error", "error_type": "error",
    "provider_error_code": "upstream_stream_read_error", "sse_event_count": 26,
    "provider_request_id": "1d9ab1ff-4896-4cf8-9c86-512896829522",
    "response_id": "resp_048c6e89491c1847016b0801da3a9087d0b9f92cea1ca055a9",
}
RELAY_887_WAVE03_CONTRACT = "9535b64e738bfcfb1479437c59e32f80207b16b3e7aaf2eeecbba4ac7c7bc26f"
RELAY_887_WAVE03_AUTH = "75879126368c97351371c13b9b952fef24e9836079097a9e6de4d5512b91dc39"
RELAY_887_WAVE04_PROOF = {
    **RELAY_887_WAVE03_PROOF,
    "previous_request_id": "89b698f474ca01c573d836049270aaba70cd41145b02ea0e3459f104f530cc50",
    "previous_terminal_file_sha256": "152fb89f5861ffe113431e3449036265517697845d2df6c3e3ef53f353cfeed5",
    "previous_transport_progress_file_sha256": "73070cee42f232570bbe4a30a2cf83630b050e35917ba4a207f7f432517b0eb7",
    "relay_not_before_unix": 1795687864.4677405 + RELAY_RETRY_SECONDS,
}
RELAY_887_WAVE03_ATTEMPT = {
    "attempt": 0, "attempt_id": "35c5eb82fb794d78a637d5c75e4be90e",
    "job_sha256": "2faf1fd49ad5967e912896851df88254a30c14466506ad9db6c24d3b41c7ca43",
    "result_sha256": "933c0890ab7376093cd62abdfb0637682a1088be8be3398a0e4fba3fea7dd0b4",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_887_WAVE04_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "0898714b-833f-46f6-a9d3-1cd9cfd0cd2c",
    "response_id": "resp_0d4ca1588c4db38d016b0804ed1f2887d2af7e53be8e31df23",
    "conflicting_response_id": "resp_0898714b833f46f6a9d31cd9cfd0cd2c",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE06_CONTRACT = "25877c1f9d94e21b6edef515716ec3131961bc584cd6e7a7d3cd293fe6e57b55"
RELAY_12203_WAVE06_AUTH = "505603229a4568db10a5d830ae602005910e37a89baee011ee5474b5b8aed760"
RELAY_12203_WAVE07_CONTRACT = "cbe9c095142e5e0ccf2f6db546b012e2a028ec19a3360df1e5030954d430dc4a"
RELAY_12203_WAVE07_AUTH = "c4c3a24bc680d92435d9c9c16637fbe31618f56c11f477a91b8124a6b5a98f54"
RELAY_12203_WAVE06_FITTED_PREFIX = {
    "round_05/source.json": "9b290ab2a6c5a7da8334eb77ca003a10ac1b52ed10a76c0baf3b02b2b9f64eac",
    "round_05/transition.json": "54470e8d33fab447b9e3704826af595f044199b0c8d2880f64a2cdc1b4cc2c2b",
    "round_04/memory.json": "0150070586759a4a2200f593df0af5207b6c63a84b92c3081996e4ddca04ce11",
}
RELAY_12203_WAVE06_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "e3f44e8d707943c3addae8000ea5c342",
     "job_sha256": "c400e12117d7d83cbf39dd2ce20e77f3a6d26206e4706eabf28d85e55e06b1b0",
     "result_sha256": "35b9107cccc252238008048eafe50b4e0de3ba0b46dad62dab281ee5f1f99bca",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "94b1558e7c0047e9b37b96e26b42e203",
     "job_sha256": "cab8059942f00cca4b9dec94e9a92e44400bdfc512f3e61b9d2f6a7272df3880",
     "result_sha256": "45b7bc50571e263baf4e2cef1312c404be6699e9986855068d82ee1681cd9ed9",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
)
RELAY_12203_WAVE06_TERMINALS = (
    "665e80f3da6b65115b03853bb3228b59df135478d7dc7b51a1a43ce5d7d18384",
    "d79bb2cf536690154f461551eb03b2781063f2b7e10dec91f6368d2dfb8ea19e",
)
RELAY_12203_WAVE06_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "6673d2374dbc481ff2d1c1ea52f734a2d4fc93d36f5b377a88b3c1afcdb7cd43",
    "attempt_00.replay.terminal.json": "a423f8707d1db0eb4b637dc9e799401908b4c026aa163814033680d4d0a95d42",
    "attempt_01.replay_link.json": "3c0326655d089be53a55e26ca0f5ebb01bbea864b3bc5461f93adcf47d3891ea",
}
RELAY_12203_WAVE07_PROOF = {
    "logical_index": 0,
    "payload_sha256": "00d53155b752dc54eed4e6aad7bfbdb01d19b21c7f0866cebd85391f8d456373",
    "previous_request_id": "980bebc61700c1aa38cea2122393f8edce7690379d71e8cedc35844317644bb2",
    "previous_terminal_file_sha256": RELAY_12203_WAVE06_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "0d98d86ef6d1088ddd53e57b2b4cb94ea5459e46271550f566acc8eb07bec856",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795688681.139692 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE07_OBSERVATION = {
    "http_status": 200, "model": "gpt-6.2-astra", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "c9fc2a09-7679-4b58-a20d-0ee33729728a",
    "response_id": "resp_0c871587f2f473db016b080824636487d0a93ee1bdab4872ac",
    "conflicting_response_id": "resp_c9fc2a0976794b58a20d0ee33729728a",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE07_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "79b3dbd9ae574275bdf5b358a6fb0d94",
     "job_sha256": "70e22b03ac3869c06f56cc9283fbf5184695e1a472b19e1b83e44b7f9d3c4ede",
     "result_sha256": "229b4d1b83302cf2471715025fec255c476bc4c01e31a17f99fa3c6da1ce9eff",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "a23ebac4293547a7bae5dd5d862ba37a",
     "job_sha256": "e3fb82f78d7c7761b4ff81c96ac50de296e06003482505d66a426877fecaef9e",
     "result_sha256": "b2371c6afe4fbbd9525ef9097025ef661546c8eb07df8256f05c4aa173168666",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
)
RELAY_12203_WAVE07_TERMINALS = (
    "e6957c8821f8fb9f7d463fa30fb58f5f84527df1f8207977b13c3856ff78c0b7",
    "40149ac3a1373cdb77683e491972ce31113f9fd8b45fe4f33bf3b36577b34ee1",
)
RELAY_12203_WAVE07_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "752e4554bd8a33e0bb475158b5f206e579d83090375afea485b6666cac9f6929",
    "attempt_00.replay.terminal.json": "b6d06b60daa1be86c15e605eb6c96a245db67b29ab021433a885741af5f815d4",
    "attempt_01.replay_link.json": "ee26381d4987507172431e0f8f3309fcbe9c4bdf717cf2877fb30eb873cfa028",
}
RELAY_12203_WAVE08_PROOF = {
    **RELAY_12203_WAVE07_PROOF,
    "previous_request_id": "3ec4ff97a04342586a18a980e0e10fdac0fb6940299263a471f4eb1b11c0874c",
    "previous_terminal_file_sha256": RELAY_12203_WAVE07_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "775ebcaab79c6f3da6f2474a79c0a68beef93b5f4a23dd4a7af24a77c0988305",
    "relay_not_before_unix": 1795690739.9466105 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE08_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 23,
    "provider_request_id": "6bb7602d-b97a-4472-b11b-35a7970d6dc9", "model": "gpt-6.2-astra",
    "response_id": "resp_0e33b0170fc4fd53016b08100f08b487d08dd9fa602b7f5aa0",
    "conflicting_response_id": "resp_6bb7602db97a4472b11b35a7970d6dc9",
    "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE08_CONTRACT = "5f65ea60af0ce2440d123a21f350e60d53de9d40e53b85557a3ec8a43aa7456a"
RELAY_12203_WAVE08_AUTH = "76fd9ea06a3de5a1b3e26c6fa0755e4ad0065b0a26841d8798f97074367397c1"
RELAY_12203_WAVE09_CONTRACT = "25793611aa80ddf279067ac651908154ecc65ad49a485b4e69c2df6e16322928"
RELAY_12203_WAVE09_AUTH = "4d19e1f3fff41d5af1f09181b8f63f5b876b92f2d5c76f16dc08ab8899d91fb8"
RELAY_12203_WAVE08_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "22736e2c597d42bba2f5730825e76a65",
     "job_sha256": "09c94cf6eae1273dfde2076230f3ec75608874c2d1c45ac80c28b152459a2ed7",
     "result_sha256": "487ce8f74f48e74f41964dd4656c4562ae192a8486b9fd6a8025d78f302fa7f3",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "f6543a336b2c4f04a3c75d585b9a7083",
     "job_sha256": "cab5a0446f3d36a6bfc8e237183b22330b71d5995c3c7e6dcd09083acc84a0a1",
     "result_sha256": "24abca31698e415c935f3030214e243c3cad42c998efd4c8dd4883b3bdb605b1",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
)
RELAY_12203_WAVE08_TERMINALS = (
    "fdca433ee02d59c560b3374f10da8e85512550ec2bf6f71db914e61aebd3ceac",
    "a0673027c885037ec65346ea56523e7268e919f7be69392171c0c346b6a8d65d",
)
RELAY_12203_WAVE08_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "968d3d51677033bf8f14283fe42d4195f9aca0f162e5f78ab852c33456788660",
    "attempt_00.replay.terminal.json": "c7c1876782e113a092c7431cce9b9269309d3c40eac074fcad8ab61ba2dd513f",
    "attempt_01.replay_link.json": "05a826f44aff6fbf994f43dcb50ce60aafe83d9af7b3f07a136f948e26f20ac3",
}
RELAY_12203_WAVE09_PROOF = {
    **RELAY_12203_WAVE08_PROOF,
    "previous_request_id": "bc13a59e4d51133e819d127814ff58e6b9d2657f5c69a1320f21aaf1d9c6a316",
    "previous_terminal_file_sha256": RELAY_12203_WAVE08_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "fb1de8ba5997deb7378cf75f47ea62533deee34fa51be135b5820dbe128fbef6",
    "relay_not_before_unix": 1795692975.7197583 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE09_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
    "provider_request_id": "9bd44948-f4f4-40f5-a811-098b71481d85", "model": "gpt-6.2-astra",
    "response_id": "resp_0736a44e75a90c12016b0818e6209087d28545d7eaea36f9c5",
    "conflicting_response_id": "resp_9bd44948f4f440f5a811098b71481d85", "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE10_CONTRACT = "168df83d138c13c153bc05e91f8ca0e73208affaf08dc3292135c269c30ec8b5"
RELAY_12203_WAVE10_AUTH = "a004554f8d14f8b75b9c40cd084b6f1dba28cf677c246abd882c191869d0934d"
RELAY_12203_WAVE09_ATTEMPT = {
    "attempt": 0, "attempt_id": "47604b0004ba4cd98dd083e22aae017f",
    "job_sha256": "8ccd651794938a33dc51fdcbfc077158f675e89d145e0d3724b69a8a2d8597e3",
    "result_sha256": "6cb112f81e78254912da39ab4539ae51fc9443308614e93b0424dadc203311cb",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE10_PROOF = {
    **RELAY_12203_WAVE09_PROOF,
    "previous_request_id": "a0bbbe0bcb9362c5ef9262f279efb9a3515d7d0407e1ae6666403f84a634bc22",
    "previous_terminal_file_sha256": "3a1d44dfccb67d510169ab721c4eca7e13dedc2fe4e6afdf28cf72c4e171ec3b",
    "previous_transport_progress_file_sha256": "33bb92f90edd5c1fd19f9ef1c1ed6bbce1e43297716e536b2f1cb469981bdd41",
    "relay_not_before_unix": 1795695678.1715395 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181},
}
RELAY_12203_WAVE10_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.done", "sse_event_count": 28, "last_sse_seconds": 980.5284760829527,
    "provider_request_id": "f870028a-25eb-4703-8571-6b0b1dec1870", "model": "gpt-6.2-astra",
    "response_id": "resp_0dc94406a9c8a9b1016b081d14b61887d285421dd0467c6173",
}
RELAY_12203_WAVE11_CONTRACT = "6083e9eb70e70d536952f4c9faa91b310c3560e6934875675f55f91d7135855d"
RELAY_12203_WAVE11_AUTH = "914a2a3187fc33a84ff6678bd7ecbab1f14aeb3692fd4d2dc1a340757d71001a"
RELAY_12203_WAVE10_ATTEMPT = {
    "attempt": 0, "attempt_id": "1f2c7f3be5294230b6a3699b9a653db0",
    "job_sha256": "eff1ba75a1cfe1657ddda50d768db4decf4c33c77be3310a39bbbf5d48db1e5b",
    "result_sha256": "759839401c4679744ad291df409d6ae9caec11bc349773245e5349be3ae6942b",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE11_PROOF = {
    **RELAY_12203_WAVE10_PROOF,
    "previous_request_id": "dbe5eb527a0236fc9e81d6714290fe8068bd747d9408362924daac4ed2ac8923",
    "previous_terminal_file_sha256": "545be789c7eb182cc8cff91c3c3e3e88b777acc2395f2f63d520e12894ae9f56",
    "previous_transport_progress_file_sha256": "1bbd1344e6ad2268c365f851f2b42c363b3d76b6620fbfb89cb7363f80c68d26",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795696851.8253946 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE11_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
    "provider_request_id": "e28495c5-d4e2-42ef-8e04-e706fae6752d", "model": "gpt-6.2-astra",
    "response_id": "resp_045a309a88ab9a96016b0827f5520887d0a8494b64e6666c61",
    "conflicting_response_id": "resp_e28495c5d4e242ef8e04e706fae6752d", "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE12_CONTRACT = "0fea4698db29545c7f8981c83c2fbb0891131832a40913d542cde8db38d24c61"
RELAY_12203_WAVE12_AUTH = "e5a55b77d5e384811cd71c158fc058eda2d139523a45ac0480f281e37d964768"
RELAY_12203_WAVE12_SOURCES = {
    **RELAY_2357_WAVE05_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/provider.py":
        "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f",
}
RELAY_12203_WAVE11_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "6f75c90ab45748eaa054c070473ec296",
     "job_sha256": "ee7626ee5b00810e19377551f121599d010da056f3f34236f322b92c3cbf49bc",
     "result_sha256": "aec0ced68b9a2daff882ad7ab3a09ac731d14055421c41d2201ac8c747e2d4d9",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "6bcb5f6a31ec42a0b5c254b787202c52",
     "job_sha256": "6d87de5c3b77b273442ca8480732588e6f45d1a57b60462794394aa8dbd92220",
     "result_sha256": "efae225e919525904cf79acdd10bf7cd2c4e076ada6e2e168c7be88250bc319f",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
)
RELAY_12203_WAVE11_TERMINALS = (
    "903b43ba21fbaabc38bbb313d6d902975dc05737b100be0102f0d0fbe2699fa2",
    "6ff9d185708909d691c4986010b25d0b8188aa096dfa21fb95ae78ddac08a874",
)
RELAY_12203_WAVE11_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "0b19a39b3fe1948be32132803058b98d417a21420ae7ac08a3978dba61fa3517",
    "attempt_00.replay.terminal.json": "ba9dda32d3025c6434533d53a1a3de0b4ecf978758fe4c6fd6cfbf84e65bc3a2",
    "attempt_01.replay_link.json": "603fb01451eca42bee560998d456bddf67d1e50a0004dcf609ea31ad33b6fd29",
}
RELAY_12203_WAVE12_PROOF = {
    **RELAY_12203_WAVE11_PROOF,
    "previous_request_id": "71b0130320deaf04a0f36006a8886756c66532c5adc43d318d0c19b23ee94878",
    "previous_terminal_file_sha256": RELAY_12203_WAVE11_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "d38fa2f6920488a978e98f9c1b6c5a9a74d9a5f44a613020e52f602b0127e77e",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795699095.9433548 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE12_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
    "provider_request_id": "0dac1eab-fa9b-4e8c-95a7-b6425611386b", "model": "gpt-6.2-astra",
    "response_id": "resp_0eecca46bad6c6e3016b0830bcf4dc87d0bf5ebeabbf62be5a",
    "conflicting_response_id": "resp_0dac1eabfa9b4e8c95a7b6425611386b", "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
}
RELAY_12203_WAVE13_CONTRACT = "bb8bb02bf7478865f6bd510c6667687b89c8044152a1af834288f4fba3125801"
RELAY_12203_WAVE13_AUTH = "68fc075c81f6c9408b99a9669a650136542f399b77573095f91d475373236055"
RELAY_12203_WAVE13_SOURCES = {
    **RELAY_12203_WAVE12_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "07d61d758c1466f27d7edc6fc38dbb718b99ce8a6449176923b21d2e92aab85e",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py":
        "6111c04c5c1eb6688e5cf6d0b086d4c1d8179a7bfcd7a9e9d04e8f572fce38de",
}
RELAY_2357_WAVE10_SOURCES = {
    **RELAY_12203_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "09d13ce1ed73f50398fc823358b5f0ee71d5e1ad857c30ccbe5eb9089d56eaee",
}
RELAY_2357_WAVE11_SOURCES = {
    **RELAY_2357_WAVE10_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7a54aa2162e685cff1b830aeb632104afb5a1fea7e0e15568720f0969eb103e2",
}
RELAY_2357_WAVE12_SOURCES = {
    **RELAY_2357_WAVE11_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "ed5e4f00bf938a913c9cb7b5c76f4973a7f32b1468f1ac21889dca813b8ec80d",
}
RELAY_2357_WAVE13_SOURCES = {
    **RELAY_2357_WAVE12_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "f046bfed18c31a7dbffaf1aad2aca23795d920094fde4302cff18cbf64aa2b92",
}

RELAY_2357_WAVE14_SOURCES = {
    **RELAY_2357_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "3d8e1cc9ba249a04f47c8926f6d62d3cfb3c05420e430ad3a2ae506cdb1f4802",
}

RELAY_2357_WAVE15_SOURCES = {
    **RELAY_2357_WAVE14_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "2645812deb783d9f1af529738128184d223d859d7d8a6ae76b677e291aa0366e",
}

RELAY_2357_WAVE16_SOURCES = {
    **RELAY_2357_WAVE15_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "fd47166c0a318e59270d336889454cf64c47819b13345d34d2485c6975eb8d45",
}

RELAY_2357_WAVE17_SOURCES = {
    **RELAY_2357_WAVE16_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "e047de29c14cd62d31c269e11df2f257608312688b5d141dee5c0054ce45fb3d",
}

RELAY_2357_WAVE18_SOURCES = {
    **RELAY_2357_WAVE17_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "8e2b5f76a9d2e713b244bfc1d0b0e860fab777a4dce5d8dd4b5d81dfa798e342",
}

RELAY_2357_WAVE19_SOURCES = {
    **RELAY_2357_WAVE18_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py":
        "5ea4b2415fc2779471990e5192d9ef9a3be1f5233b20b42f651c034f6d493774",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/dispatcher.py":
        "073fe84d02496a4e6ca8187922df58399d87e40b062d0521a925716fe84d576b",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/provider.py":
        "4b3f0f181401f03f3b54bc8a35861fa91a40a497c93a0162fbb7f65dd2ffa443",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "98a675ba0f0ff4637d3635c4092de8209ec59aab836d8a995cb9e1259caad33b",
}

RELAY_2357_WAVE20_SOURCES = {
    **RELAY_2357_WAVE19_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "f3152456cce4f7f92b1b7f68ec38d6fe373cb1b34b01e6b17090bf33a6a35187",
}

RELAY_2357_WAVE21_SOURCES = {
    **RELAY_2357_WAVE20_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "95e07a3659ccbb4247171801859a1e1a64444a4e19b44d496e35c7281ba8ca30",
}

RELAY_2357_WAVE22_SOURCES = {
    **RELAY_2357_WAVE21_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "306fe482e686862e6f07494a082fc59221621ab6bbd7e6aa37fc03c0ac90f3c9",
}

RELAY_2357_WAVE23_SOURCES = {
    **RELAY_2357_WAVE22_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "74140acca4d426c7c091cf7e2295d57523bf7f3c55117fabd2375d2419f1f555",
}

RELAY_2357_WAVE24_SOURCES = {
    **RELAY_2357_WAVE23_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "63ba9320a1740b306c3b5f8fe930ee1284e0d4d1865c7e93a9906cebcf09c69d",
}

RELAY_2357_WAVE25_SOURCES = {
    **RELAY_2357_WAVE24_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "d06d90fff881eee61474033ad2da4ec1db21738f86f5419d7907758eba843d39",
}

RELAY_2357_WAVE26_SOURCES = {
    **RELAY_2357_WAVE25_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "95a2c59314d7f17f53cf6c79a3d1d038b21a658c6a9c600560d599c3e8d5493f",
}

RELAY_2357_WAVE27_SOURCES = {
    **RELAY_2357_WAVE26_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "8f53ff1c0c74e3cdb5494fd3ec8e103e7c4fa4a22f8442a16326924d8fe0fef3",
}

RELAY_2357_WAVE28_SOURCES = {
    **RELAY_2357_WAVE27_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "efaff6386430a421040ed49c055d00d6a2d7bd24b33bbb5fb746c359c56c5b74",
}

RELAY_2357_WAVE29_SOURCES = {
    **RELAY_2357_WAVE28_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "8980b85724545acf04a406151eea5064e5bceb463ffd64c645911c8228004a4f",
}

RELAY_2357_WAVE30_SOURCES = {
    **RELAY_2357_WAVE29_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "c83c2d294a63087ce00b269406401bba385b3e8a6b4944dca9d98ae03364a734",
}

RELAY_2357_WAVE31_SOURCES = {
    **RELAY_2357_WAVE30_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "81f2bdcbcd0e50662e6376fdb171583c27667eea99749739494b00e2f8a09b51",
}
RELAY_12203_WAVE12_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "7b8dd2a160914ff293622380dd1d8cae",
     "job_sha256": "402f872ecbfdb2cc509973c39da844f5518d278fbe25971fa14f767f0c298e6c",
     "result_sha256": "9ccf002474dc31a8fdb3176655b390b6ca57823943b8062132ac9145d30516a4",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "ecb84ce8e8214cca9bb3005c6ff672e0",
     "job_sha256": "b941ade623118f585b2e828a8b513f290055b25ec4b8ad34564e1625d6acc4f0",
     "result_sha256": "9ca6e60ea32cb8b58d67305f78dd213dbdeb7e1bb4b7a4e4fb5c093183211537",
     "error": "connect_error", "provider_dispatched": True, "ambiguous": False},
)
RELAY_12203_WAVE12_TERMINALS = (
    "9da7c12c45a23332ed86f2d208f352192e53108f95de8f32bfe4102e56829cf2",
    "19a85316433270504a4c3c5916971129fe37adca81709c67d1274ee432b7307a",
)
RELAY_12203_WAVE12_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "fe3f21744a668027bcce37fff1337384e31257d028242f19d5154751a6bfe278",
    "attempt_00.replay.terminal.json": "54a9cf6654cba8bc2e38b49948ff022b061731daad02367c25959ad4b2a8886a",
    "attempt_01.replay_link.json": "dd8842bef86d5602e5e2d0d8eac25e844b07f05ad12bdcd8ee83feaea6ca7775",
}
RELAY_12203_WAVE13_PROOF = {
    **RELAY_12203_WAVE10_PROOF,
    "previous_request_id": "0f9446bd6febf0fcda32cfcddb70af5ecd7e097a4cbaed4071c0fba504482860",
    "previous_terminal_file_sha256": RELAY_12203_WAVE12_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "299590b71b3684af2877a6bae97f32473e835d432a054dc455dbb34cbc15c0c9",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795701276.67948 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE12_POSTED_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 23,
    "provider_request_id": "470d4f0a-1b3a-4ed0-8460-d71f12a292a8", "model": "gpt-6.2-astra",
    "response_id": "resp_065c53b252d45ce2016b0837cab6b887d0b6ff67a016f45b7c",
    "conflicting_response_id": "resp_470d4f0a1b3a4ed08460d71f12a292a8", "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error",
}
RELAY_12203_WAVE13_OBSERVATION = {
    "phase": "transport_failure", "error_type": "connect_timeout_before_post", "event_type": "ipv4_preflight_http_401",
}
RELAY_12203_WAVE14_CONTRACT = "158dcfc081b4f75dc2aabf6296a1d07440fe3680ab278b70c5aa51adec633ef0"
RELAY_12203_WAVE14_AUTH = "443fba215aebbbc0e6170e6d2b9ab8b37c8298e795d2d3e7332333ce5f0e2adf"
RELAY_12203_WAVE14_SOURCES = {
    **RELAY_12203_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7df1bd70c3dc6c8166fed8167d4603281182d7a39f23091decf903a4999cf7d9",
}
RELAY_12203_WAVE13_ATTEMPT = {
    "attempt": 0, "attempt_id": "51559c40c09344f1a05f16bae486785f",
    "job_sha256": "c52a75b11fe8b8315c73d6322434c70c32b5302ac627dbc7046188e2745bb6dd",
    "result_sha256": "ccfef94b6a6b867c063d7e4907221ce70db3cc064e432da7dc73e46e0e859bab",
    "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE14_PROOF = {
    **RELAY_12203_WAVE12_PROOF,
    "previous_request_id": "094821a5d94e07fd86e2a799bc29905037d405f924048b116eea204a5c398962",
    "previous_terminal_file_sha256": "f5254ed9b334038058004aa91a1861b99b91d59ee25d2790890bbd3ac6b5a0be",
    "previous_transport_progress_file_sha256": "fe1c0ffa23209796269b9f47e1e5a5d281a59229b81132f999da226f5bef7f77",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795702290.7997937 + RELAY_RETRY_SECONDS,
}
RELAY_12203_WAVE14_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
    "provider_request_id": "637102d2-1bbe-40b4-952b-88f7fd4b3dd8", "model": "gpt-6.2-astra",
    "response_id": "resp_088293b7705ca974016b083d3a738887d0b2d4e01c94eb52cf",
    "conflicting_response_id": "resp_637102d21bbe40b4952b88f7fd4b3dd8", "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error",
}
RELAY_12203_WAVE15_CONTRACT = "30941667c357c8f3e11342f7e6044629721d060a7b6b8c0e5d1933f0b16c75d1"
RELAY_12203_WAVE15_AUTH = "846fc429706ceff7454ecb653b171683262a989a2e664480c0954ef7d521eb30"
RELAY_12203_WAVE15_CHECKPOINT = "34cf4e966602c4eed13ea70922aa80313ad415dd5b3cefe7fd2e7650f05a4c07"
RELAY_12203_WAVE15_SOURCES = {
    **RELAY_12203_WAVE14_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7ba430a5d284c78acf8ca57785ffc0906f30b6c4927598fce92b7683780f8b2b",
}
RELAY_12203_WAVE14_ATTEMPT = {
    "attempt": 0, "attempt_id": "fdb4baee7c5b4f0e90eb85782b464612",
    "job_sha256": "dff0b37c70e9cbbb39d3b89d031f5e1529b329a7692b002a6caf15524c7bc093",
    "result_sha256": "97ae0af3ca4b8fc33467947e299d99c8afff6d432874e8c31f1784dd83f6f380",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE15_PROOF = {
    **RELAY_12203_WAVE14_PROOF,
    "previous_request_id": "529d87bb7b93bcfea2b85466ecaafecaf802549c63ef9eb23cce67ea4ce0b17b",
    "previous_terminal_file_sha256": "70fc3592a2dd242f1fa7d5a105ccd4ff965a181c2badc379a27bf5863d4e2f6e",
    "previous_transport_progress_file_sha256": "ba07dcbd07341c90a9e971277f49f4ce116e6c7c171b5b2167425c152c8cbc89",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795705935.3553925 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181},
}
RELAY_12203_WAVE15_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.added", "sse_event_count": 15, "last_sse_seconds": 153.12645345786586,
    "provider_request_id": "9309d0ca-9a4f-4459-a16b-9c2a037f642e", "model": "gpt-6.2-astra",
    "response_id": "resp_002c70521cd2c530016b0848517b9887d0a373425c8bb727f2",
}
RELAY_12203_WAVE16_CONTRACT = "a6bfa7d6fa6c82cdf000aa859699ea37fe80c970fc762b0ea22dba545d4a5aa7"
RELAY_12203_WAVE16_AUTH = "f49406db892e95b4e85f2d6fd5061238fd42b6fdf72501ffc167e47861bbc02b"
RELAY_12203_WAVE16_SOURCES = {
    **RELAY_12203_WAVE15_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "e815a92283e9609dec698382751ffa71a0fb1651968e90bbe877b828d4ef46da",
}
RELAY_12203_WAVE16_PROOF = {
    **RELAY_12203_WAVE15_PROOF,
    "previous_request_id": "3ea5066c06c0fe613d6b1d312734aab65e9f38f0224dbe04d10e99537ff50ee9",
    "previous_terminal_file_sha256": "283c19e27fd67795e3a53aec8f012028aca447615fb2276d7735047727fffa74",
    "previous_transport_progress_file_sha256": "2258c99051d47806e1f6b54f48d1e7289fedcb77b1a1a865e7881db1f6ff43c6",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795707966.662386,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE15_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "4fdcc2c2a5b9469eb0e0eb0124b7ad0e",
    "job_sha256": "4426c7dc986c68654f3ee8976fb1fe80a0db9821ecd5249e399c531ec13591a0",
    "result_sha256": "be11cf9bedb53a7b00a71eb263cf98783714c0259e6e63c02429fc6868c8765e",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_12203_WAVE16_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.done",
    "sse_event_count": 8,
    "last_sse_seconds": 568.9974320828915,
    "provider_request_id": "2cf1bb57-c0c2-4841-adc8-36cfcb7c15b2",
    "model": "gpt-6.2-astra",
    "response_id": "resp_005ad517c01224af016b084de4458887d0a9fa543067dee748",
}
RELAY_12203_WAVE17_CONTRACT = "9d68d3cd05cdfb725c5cfbad68474154748bcff75f06b7cf40c9463980c8921a"
RELAY_12203_WAVE17_AUTH = "211b469f82f130d714bc3ccdaf8ab3ae3d116237303b81918bf13c6cb33419ba"
RELAY_12203_WAVE17_SOURCES = {
    **RELAY_12203_WAVE16_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "853731b40b0c2d55249355508ffabb48794278ba68dbbddc52b7cf6c2cdb6c6d",
}
RELAY_12203_WAVE17_PROOF = {
    **RELAY_12203_WAVE16_PROOF,
    "previous_request_id": "6c6b7c124f80b82a41809003d7fd7b3b2dc6b8fb3bc3e738b1daaa3e24d1624c",
    "previous_terminal_file_sha256": "d408a77082c3af6bdca7dd6d253a2a263d280e9b8257f12d15f7d024e6192ad0",
    "previous_transport_progress_file_sha256": "e7b7ea9a15ec61d758bc1c7690b9504050d809dffc60e345e076b9d12584e116",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795709911.0657685,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181},
}
RELAY_12203_WAVE16_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "2b0e526441584843bac2a9a185a5f4fe",
    "job_sha256": "c42744c3574b7c569850e860b0341ab331fd64cb2c222937f7f0d75197a10539",
    "result_sha256": "763787be79a67319d1da98ba84644d7d12f9bc5e33478fea5f97cd30dc424678",
    "error": "connection_interrupted",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_12203_WAVE17_OBSERVATION = {
    "http_status": None,
    "phase": "transport_failure",
    "error_type": "ReadError",
    "event_type": "ipv4_preflight_http_401",
    "provider_request_id": None,
    "model": None,
    "response_id": None,
    "sse_event_count": None,
}
RELAY_12203_WAVE18_CONTRACT = "63922d96bc95ece68bdb8b2789370d20966e9a6cfd60dde903f6c4d3c1f5a37f"
RELAY_12203_WAVE18_AUTH = "4539f6fe8b51e1595907e023adf0e12de638c7c94d433bc5f58ce6493879fa0a"
RELAY_12203_WAVE18_SOURCES = {
    **RELAY_12203_WAVE17_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7b4b31c20da853f485fc71c11eb8eb7ff5fa21c8c97bd1be0c381e52fe49bb21",
}
RELAY_12203_WAVE18_PROOF = {
    **RELAY_12203_WAVE17_PROOF,
    "previous_request_id": "b67d3712badb2793ef1d60eb765532df03c73018896561bebc491d58d7984b93",
    "previous_terminal_file_sha256": "22ed13565ffdbaf4138e5d320e7f00f47439c3b255fa82ba4366f80a8896d289",
    "previous_transport_progress_file_sha256": "2dc92d87f3e6cc06b8cf9960c443170a9098a49d88b469bd3f5de183631e7fd2",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795711982.841558,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE17_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "3ef3bfee813c47acb717ba512634e87a",
    "job_sha256": "2992c3d73abfc1851e774777200b3d9f8e5db41adaa7613281e31c9ccc88a57f",
    "result_sha256": "914807619b3dcfa3d2ef9c04a6232a08e135708b9f314960a0d72a71e0135eed",
    "error": "response_id_mismatch",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_12203_WAVE18_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "last_sse_seconds": 211.33261599997059,
    "provider_request_id": "020886eb-e25b-468a-98e8-a5b6da9b33f9",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0eca49db69bee932016b08625d14ec87d28bb2cef15a58ae8f",
    "conflicting_response_id": "resp_020886ebe25b468a98e8a5b6da9b33f9",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}
RELAY_12203_WAVE19_CONTRACT = "49bb4ef1565ed0ae87249701e9079728d6945371a43ca230814d6d54b57f3f43"
RELAY_12203_WAVE19_AUTH = "77b2d1e58c7a53b7a410a1906222b495b752787328af7d83f7a024e30f29fc94"
RELAY_12203_WAVE19_PROOF = {
    **RELAY_12203_WAVE18_PROOF,
    "previous_request_id": "311a66cacc49616d2d96466027ea69d5663e28190a7d0b8d6d089fb55264f26a",
    "previous_terminal_file_sha256": "dcaa7599809f0e2971402549917dfce8da3d65aa3c9a0561b20cbacc9dbee176",
    "previous_transport_progress_file_sha256": "e3aa3de2fff7e7de647f48119e9e559f9a7244dd31bcbe8eb447822fffe86360",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795715097.1493309,
}
RELAY_12203_WAVE18_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "0d93c0e153a34a7b8463363771c657c1",
        "job_sha256": "565d4b8e3d2779a85bd55be27ba770ddcf982eb29d58096aba2d84291091c02c",
        "result_sha256": "fe7a3230fa1739c0e691e831fca0fcd26b240fc0fb5f994062580ea24db72221",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "d95e1a4f686a466ebff67c01c09bdd9f",
        "job_sha256": "1729d2b6cb97884274cef553d3917ff24416da6e00a906a50fa99e90ebfe89c5",
        "result_sha256": "ccaea91b793a3a7e32afe7d2410df226ca6c2522353446ff83487340ffcd5484",
        "error": "timeout",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE18_TERMINALS = (
    "a0aa0b391806467a3b91e2e14b25e2244e0e1a77e46c3a310f7837cce36b34fd",
    "dcaa7599809f0e2971402549917dfce8da3d65aa3c9a0561b20cbacc9dbee176",
)
RELAY_12203_WAVE18_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "97066490432870b2c57a338195df9329d3f918966ec82e15df4faaec9f25e655",
    "attempt_00.replay_authorization.json": "c6519b97bc46d29e6bb21a05b66fa34a9eea6194fad5b1516a6e25ce6e1f505a",
    "attempt_01.replay_link.json": "1c91a704844194e69d091d9a5b3db1f3235d026765a2f5aa42807db2e6e71a7d",
}
RELAY_12203_WAVE19_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.done",
    "sse_event_count": 40,
    "last_sse_seconds": 601.0424845840316,
    "last_sse_activity_seconds": 601.0424845840316,
    "elapsed_seconds": 1501.0500557499472,
    "provider_request_id": "0308e63a-cc9d-4443-ac78-ada08c120344",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0a6f786002d26b85016b08698f43fc87d2977526038f2f2c8f",
}

RELAY_12203_WAVE20_CONTRACT = "a3eec33906e719e0c25a5a070411640d5ff1b00a99738484a66b5fdaea833d55"
RELAY_12203_WAVE20_AUTH = "901f6ad61d59f658a401cd7acccd0612a3c15e3b4c789562603a50e9e4b64f16"
RELAY_12203_WAVE20_PROOF = {
    **RELAY_12203_WAVE19_PROOF,
    "previous_request_id": "80240ab888f288dec5cbf7b4abec27cb48e7d47615902c72e3a6512f07079ad7",
    "previous_terminal_file_sha256": "37c1c267b7aa48726c180d910d828f61475b243024a7a7e40fa349855a6661c4",
    "previous_transport_progress_file_sha256": "a9d63d20ec106348db4c2770be209270da92d99a837c789753fb851aa49d87cc",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795717790.025714,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE19_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "b57d782d8ffd45edb332d4dd612e0197",
        "job_sha256": "ed465796ba7dda913c4b22f901b5f09e1e40a7f18c22c1a1b459ab89182f1964",
        "result_sha256": "9a51721e86002bce279451db544d4f044dd97b38e00396e1213bd810e4285e80",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "7fa11bb380bd4d409f4f85c69df814b3",
        "job_sha256": "d4c2208ae015ce72d14b466b481f1f909ceab81b38b7099f8bccd676ff8a35f8",
        "result_sha256": "1918e76accd66ae2e30b0c6556ba94601e9add98713d0a9230bee88246df82b0",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE19_TERMINALS = (
    "6c1b0b897e9505bb575d167458e220d116c80f7228cd173972750c2799ccf8f1",
    "37c1c267b7aa48726c180d910d828f61475b243024a7a7e40fa349855a6661c4",
)
RELAY_12203_WAVE19_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "5b716a384dde6aaec7d77c95f504547f701271ba66f760e0cee06cba0f4212a6",
    "attempt_00.replay_authorization.json": "2ea77975157a68fd1d1f609420b6b389f8769263be4dcaa03eb6f52003246b18",
    "attempt_01.replay_link.json": "a5a2c9081e7519df6901789588902cf128888c8e8761494a4d2ad97f04a4d4a1",
}
RELAY_12203_WAVE20_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "provider_request_id": "94641705-dd6f-4d42-a2c7-f13b78fd24c4",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0758128f78ca6ba4016b08791cd33087d2830e3fd378fe2c3f",
    "conflicting_response_id": "resp_94641705dd6f4d42a2c7f13b78fd24c4",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_12203_WAVE21_CONTRACT = "70b7fb578ac41d9aac9bc36d651057ef61fd234398c73b207d97dbc469649033"
RELAY_12203_WAVE21_AUTH = "1a8d3d7e6719e79441ca3e2fe0c01054ea378af1ca63446a7b6e39d42a51dce3"
RELAY_12203_WAVE21_PROOF = {
    **RELAY_12203_WAVE20_PROOF,
    "previous_request_id": "ba765e1b6d6bd90cab93e314cbda5dd40463e06708c6123be8674c25c1ed4d5d",
    "previous_terminal_file_sha256": "2e3f94479dc7028ed018fc82e7cf4964e4f02ec134887082a5758a3e14576cb3",
    "previous_transport_progress_file_sha256": "1c41810acd031369902531dd1a22b6bd3fcbe13a297cd2896940ba2f6efc777c",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795718946.3791935,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE20_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "c234a16437b24f02aeb25f9880152e66",
        "job_sha256": "dbb6e4cefa02fd600e4c3783736d00edc98a6c5e6dd626931c4955bdcaddd088",
        "result_sha256": "3bd7f01a20c518f20317b3f81ede50c3cd2856a42a6f9d311241035128867830",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "2d6754e26cc847dcb6c4dba057646ecd",
        "job_sha256": "01f4db72e3d15371b541f444193332587d689fcdcba082d3fc4e676fef84a50e",
        "result_sha256": "116b5bb4990ccc67e0d804c751ad2429e77b200908dc1c409389140d19e6c84f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE20_TERMINALS = (
    "3411c110316f7facfb777f875535a93372a7490c9764c891f271d5e3d480b87f",
    "2e3f94479dc7028ed018fc82e7cf4964e4f02ec134887082a5758a3e14576cb3",
)
RELAY_12203_WAVE20_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "7d641c0c581a91638a1d517ef9e8028bd4f826f19398bf79585d383f416b2520",
    "attempt_00.replay_authorization.json": "0962610faf4ae8efad79e4477bd8ef16604ef4bac6b4fc20162a3b9dce3f4794",
    "attempt_01.replay_link.json": "b15b845a350c55ced28d0d1f08c1f263828a7a72dfa8797031f0bec47d44c38a",
}
RELAY_12203_WAVE21_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "9140242f-2d65-4f15-901a-f0d8fc0ef8e3",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0515e086da96efaf016b087d952d9087d18f4843fb7407898e",
    "conflicting_response_id": "resp_9140242f2d654f15901af0d8fc0ef8e3",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_12203_WAVE22_CONTRACT = "e885e39585f247f0a629c94d5e2ef77debdd441e66e2d3eb24f23eaa3b838686"
RELAY_12203_WAVE22_AUTH = "9c7ad4e88455e2d0daa75ef3ed9b90a93bf17d1484327715bd543608e6f70ed3"
RELAY_12203_WAVE22_PROOF = {
    **RELAY_12203_WAVE21_PROOF,
    "previous_request_id": "473e3e929c6548cff9b2f6391a35e7bec82570feb4ce781b770c5273c8b3fa31",
    "previous_terminal_file_sha256": "fa5729da7f1920987643fcc497f0a1e47e3d056b5669cae0e29d8dfdf82ab24d",
    "previous_transport_progress_file_sha256": "147b8c428f6ba4666ae0c8155773f1bc9b972e81858f2a931ed34c3c42eb48ee",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795719938.7627635,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE21_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "0dd696b221124af1b6c7b8e9c2c801fd",
        "job_sha256": "9c77ce93efb842b5df0ad2f6ae18a8655709f92aaff116afcf39850b786eb0ca",
        "result_sha256": "074f3aa259ae089e25009f6ba1d109600b8f5c2486a77175e8599133ef0cb90e",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "53851a8fce95464fbadddeb5b692cf7a",
        "job_sha256": "a7b34485b706fee8b1cf5111f2a640516f43b959a47dc368d8a03efd5418376e",
        "result_sha256": "7c19370a27531331a705657cfa05b550f9c88f47404b2dd6bbb774bc64a93223",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE21_TERMINALS = (
    "c90e7e2da595681f809b2fc97f7e89b0c65e974282f3106f7f3d93bca7c94534",
    "fa5729da7f1920987643fcc497f0a1e47e3d056b5669cae0e29d8dfdf82ab24d",
)
RELAY_12203_WAVE21_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "4e034216f36a394bac49a2fe1b3631d5f4fdadfcf0cdc29a54bb9f470b2e1877",
    "attempt_00.replay_authorization.json": "f0012e47ceef357718497542e59fc68cc6b00f598df51790680dffebe4bd1891",
    "attempt_01.replay_link.json": "50812e27152b546a898b76eacd26ec6805a324780433fcc993b938ab69a66422",
}
RELAY_12203_WAVE22_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "fd79d365-c0a9-476e-af80-267871d433b1",
    "response_id": "resp_00cffacbbc637616016b08818304bc87d2baca34bef23ebeef",
    "conflicting_response_id": "resp_fd79d365c0a9476eaf80267871d433b1",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.87378316605464,
    "elapsed_seconds": 199.54550095787272,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.58489670790732, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.587101500015706,
                          "interval_seconds": 0.002204792108386755},
 "response.output_item.added": {"elapsed_seconds": 19.589333874871954,
                                "interval_seconds": 0.002232374856248498},
 "response.output_item.done": {"elapsed_seconds": 28.865945207886398,
                               "interval_seconds": 9.276611333014444}},
}

RELAY_12203_WAVE23_CONTRACT = "924ba82ee83a5280b2d10daa5f750e60521a7c88a4375c22ebdd767d669fc730"
RELAY_12203_WAVE23_AUTH = "b95ee9dfd7dcbad3c63a85b996baa5ff5588956e94c4c11124eb4530bbc252e0"
RELAY_12203_WAVE23_PROOF = {
    **RELAY_12203_WAVE22_PROOF,
    "previous_request_id": "3ac31ab1eeb9cddc94cc7ce1962c1981d6ca79e23c7ee1dc7ca03b217bd1a2e7",
    "previous_terminal_file_sha256": "db1ee53749d5ac595233d9c1efe14a91971e2cd3ca3f26ed7cf837a3fb655c77",
    "previous_transport_progress_file_sha256": "a4a35e2a988b993bc530f3c46778e70f6a14f7539272030129ad56e0ebec1a94",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795722137.2408683,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE22_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "5c8e1aa9a3e2468489d5e2f17c9cd061",
        "job_sha256": "940bae9dbc20aff9334fd4b054ac20b8ad698b1cec672b05004320d8b2677959",
        "result_sha256": "9662c638c5c230381125a3156e8d93659f90af3edf35a34706c1d0730bad5fe3",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "df7c651d002142829aea8c19d2be8af3",
        "job_sha256": "7e3f33772b9ac8ea236ba9b41864f8c3d5114e4123063c384de32826205d2c4b",
        "result_sha256": "4122a448f580ea72877de7680f72e3957d881069be8e3ae1f50ee61c46f6bb42",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE22_TERMINALS = (
    "8a18c7fa771db6f429a3e1cb27afdd9f6087f6ae3b2ff8a15e9e49a441292de8",
    "db1ee53749d5ac595233d9c1efe14a91971e2cd3ca3f26ed7cf837a3fb655c77",
)
RELAY_12203_WAVE22_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "3c80dcf3735537b8389b91c9f1364309dce997eda2a2ab83a308b723c8a2d80a",
    "attempt_00.replay_authorization.json": "e889574ff2dc535f931a57b27a3b1a51f4bb54b05efb30b06b68f3e35145edc3",
    "attempt_01.replay_link.json": "7abd6c22e2d4b485dd8bc02bb2efe3229d1964489b6f81c6747e94e3518d2f92",
}
RELAY_12203_WAVE23_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b39bc58f-4bf3-4457-b0ea-d92f4b8b2cf4",
    "response_id": "resp_071b06bd17676762016b088a08fe4c87d18f1b3c7bac30eb66",
    "conflicting_response_id": "resp_b39bc58f4bf34457b0ead92f4b8b2cf4",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 201.229527708143,
    "elapsed_seconds": 210.57383850007318,
    "sse_milestones": {"response.created": {"elapsed_seconds": 30.770994333084673,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 30.77340200007893,
                          "interval_seconds": 0.0024076669942587614},
 "response.output_item.added": {"elapsed_seconds": 30.77498145820573,
                                "interval_seconds": 0.0015794581267982721},
 "response.output_item.done": {"elapsed_seconds": 46.84271379117854,
                               "interval_seconds": 16.06773233297281}},
}

RELAY_12203_WAVE24_CONTRACT = "ebb25c05bda831640eecaf6f7029d04a725c5d0da898b442b179288907e6f791"
RELAY_12203_WAVE24_AUTH = "5720b107ac7a29a6a9dab47acbbf3627ea3e0ed278d1b330f4b7fbcaa102030b"
RELAY_12203_WAVE24_PROOF = {
    **RELAY_12203_WAVE23_PROOF,
    "previous_request_id": "f5ab73402adf522c9661d384c0d0ece42bf9169f6d4515228e78c1663e1d0e4b",
    "previous_terminal_file_sha256": "88fd1e2c33ac0d6b3e3c47841cd60e1393c9fc6d8a709efeb72996a6d71a9547",
    "previous_transport_progress_file_sha256": "8391ae3f4fbc5aa4f697fab9de3b7d97cf889eb4ab2779ac0f25cdf7f8e6e53b",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795723382.4180803,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE23_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "e7dc3a17644e4204bd63ee32ca8b819c",
        "job_sha256": "adc59e603656e8e4bc32d043b6dd41d1300f9ecf6689c3e8ce381ed0f732e825",
        "result_sha256": "db0b86745315d5d93a01f2e30e777d9bdf6657692ac3d2569b1ccabb8453a450",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "d50d43f67cbb4f0f814baa9122654bea",
        "job_sha256": "b4a2b6621932cb7662ea8f35bfece28e9759e9325eaa1df55640190177c3f32e",
        "result_sha256": "f2d7c318bc1b86345dcc40c4b9919d4749b91def1028736182ff899b44f1925a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE23_TERMINALS = (
    "f612246bc9f3fae685948de3d4e64bda72cd8d728b8dd97a781415c0696a6a33",
    "88fd1e2c33ac0d6b3e3c47841cd60e1393c9fc6d8a709efeb72996a6d71a9547",
)
RELAY_12203_WAVE23_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "0ab8e8984dca2f573059713760d30000c466c873ef881b2d26f48e03a2604752",
    "attempt_00.replay_authorization.json": "ead692bafdaee8808fb994e94e67656decfd617ec0235c6ed5d91967033dcad9",
    "attempt_01.replay_link.json": "9ba2232f2cd895cbb2438d0d7c0130b58c6ec6430948814365dbca75e0d47565",
}
RELAY_12203_WAVE24_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "6da7f03c-5c15-41d4-bf62-c23f0364062f",
    "response_id": "resp_09e3d709ff4cb791016b088ef0e73487d1900550222ebb0039",
    "conflicting_response_id": "resp_6da7f03c5c1541d4bf62c23f0364062f",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.0682915840298,
    "elapsed_seconds": 197.92816583416425,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.92737291706726, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.929158250102773,
                          "interval_seconds": 0.0017853330355137587},
 "response.output_item.added": {"elapsed_seconds": 17.931698334170505,
                                "interval_seconds": 0.0025400840677320957},
 "response.output_item.done": {"elapsed_seconds": 27.289097334025428,
                               "interval_seconds": 9.357398999854922}},
}

RELAY_12203_WAVE25_CONTRACT = "2bbf4ef9605cb96197f5bec605bc48ff066aad987d630867511eaedbd812051f"
RELAY_12203_WAVE25_AUTH = "2e0b9389fc5e539d09a480bc499c4baa3eb8ced57c99c0d4260f1f01205762eb"
RELAY_12203_WAVE25_PROOF = {
    **RELAY_12203_WAVE24_PROOF,
    "previous_request_id": "e40f78bc0bb14345f3c96ff1706a31c393692a55f8452d4bca3fedc3cf3905bf",
    "previous_terminal_file_sha256": "47042d1223569116bad111564a194f3b417c38aa615ae4bd282bde263c121a9b",
    "previous_transport_progress_file_sha256": "5d969fe6ce45782e175f9f21e4e932be0f59a08fe7d376bcbc5d1e4c14505e68",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795725286.1941266,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE24_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "8d53189d20c74d69a5b9ebb3729ac285",
        "job_sha256": "abbba70b470f41bb4b87af3d566da910a7eb6e9924dc8c02e30e163614a130c3",
        "result_sha256": "5c307ddd4a9bd05c8e8e1f4e32932c25039462ea48cae069a0aacf40e67e76c0",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "b31f16e0b5204612add655f1995cc71e",
        "job_sha256": "47c0a777186cbf3a4a2dc3c43170df1af08b124775d1108681eedbb381c2b0cc",
        "result_sha256": "5b6988c729e2d94a4d17170acffaf00c075487588745606108d50f4f96a564c1",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE24_TERMINALS = (
    "55a3705a916a01da5ceccc283776377339466e9a2d8c564aa1f2c7e2a8f61124",
    "47042d1223569116bad111564a194f3b417c38aa615ae4bd282bde263c121a9b",
)
RELAY_12203_WAVE24_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "c01400967a6ec414b28399074db948d21386a2a78e6de9afeee4be5af1a601fc",
    "attempt_00.replay_authorization.json": "9ad1c6af3e496846554c3a4a0792dac068834c20483ccefa4f4e8cc7f9cfb88d",
    "attempt_01.replay_link.json": "93e5218bb272f42a304a5ca2ff3f928b11648450259ce65c6142f2fdae3a5578",
}
RELAY_12203_WAVE25_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "86033026-fd7b-492b-9a50-f37a64ece320",
    "response_id": "resp_057f1d06095cf713016b089664e6d487d1800531f4e5cadf57",
    "conflicting_response_id": "resp_86033026fd7b492b9a50f37a64ece320",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.96779104112647,
    "last_sse_activity_seconds": 197.96779104112647,
    "elapsed_seconds": 200.7695372500457,
    "sse_milestones": {"response.created": {"elapsed_seconds": 20.84528554114513, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 20.84716658294201,
                          "interval_seconds": 0.0018810417968779802},
 "response.output_item.added": {"elapsed_seconds": 20.848750666016713,
                                "interval_seconds": 0.0015840830747038126},
 "response.output_item.done": {"elapsed_seconds": 30.206162833143026,
                               "interval_seconds": 9.357412167126313}},
}

RELAY_12203_WAVE26_CONTRACT = "670ed3f7438fe0e02ec0bdec3deb08f23cd5ed4841a2f56f19a61b745f483a2c"
RELAY_12203_WAVE26_AUTH = "92ec7ceefa9e7d81c9e7ffc4a8421e3f528eba5a8d396be92f57530f3c1c7e70"
RELAY_12203_WAVE26_PROOF = {
    **RELAY_12203_WAVE25_PROOF,
    "previous_request_id": "97980a219599e17c10f77e905b59e62515bb21ac46ee0182faae2d6f959f0dcc",
    "previous_terminal_file_sha256": "747c239aa8a093bc86a3776c6a76e084dc74666414df497052c973acb305a097",
    "previous_transport_progress_file_sha256": "98b3a95c6997af60b42cf3230547354c2075632719f9cc5af837bd65dbda7759",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795726373.246844,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE25_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "e6f1d4b3ee7145cc91dbd2970967f83a",
        "job_sha256": "07d623855102fab8bf4252d55115957313cc185e734615fc3d61193e6c7ecf95",
        "result_sha256": "afe709092017a41347f83b02677c5e9ce80bc2a610cd4cdbe03e158ce8efe14b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "ed84d3db2fe14fb08ceeee420241e445",
        "job_sha256": "12bd4313d1c87e06136ff48e6494b59ec0ef28997263d8dcee40578b0cb5c0ff",
        "result_sha256": "f07a8646b7c797459c7bb905e0c09fe4450179272b117f0ad334d7a38186b467",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE25_TERMINALS = (
    "21780831e37426a092e2b5aea1533fd4facc1a0bdee362f91612450e6f70a4c0",
    "747c239aa8a093bc86a3776c6a76e084dc74666414df497052c973acb305a097",
)
RELAY_12203_WAVE25_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "2830a89082370f6d311a2eb43a95f0475a92e6c427fa5a69b2666de7f4446bbe",
    "attempt_00.replay_authorization.json": "d7569b90c3b89161f41093704cfc1a9c56133f6d0db1c445846353b65c1cd6b5",
    "attempt_01.replay_link.json": "dda02620b82c8f2321f0b5d9f0141bce9e61bf36776537b598ec3d2087d20374",
}
RELAY_12203_WAVE26_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "4b176877-fef6-43a4-9976-a36ee210a947",
    "response_id": "resp_0725e99f1a22f3b0016b089a9a8e9887d2a5d8c1aa0e810611",
    "conflicting_response_id": "resp_4b176877fef643a49976a36ee210a947",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 204.1329087088816,
    "last_sse_activity_seconds": 204.1329087088816,
    "elapsed_seconds": 210.04968970897608,
    "sse_milestones": {"response.created": {"elapsed_seconds": 30.04070670902729, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 30.04346466693096,
                          "interval_seconds": 0.0027579579036682844},
 "response.output_item.added": {"elapsed_seconds": 30.045470041921362,
                                "interval_seconds": 0.002005374990403652},
 "response.output_item.done": {"elapsed_seconds": 47.67604554188438,
                               "interval_seconds": 17.630575499963015}},
}

RELAY_12203_WAVE27_CONTRACT = "44fcaaafc7c8083456cb7fb20105ccb81f5b8b63eba0defd77d3c1660ee5131f"
RELAY_12203_WAVE27_AUTH = "a1b4a0e28d1758ebdfe2037d0895377706de93aa0eec8b071535e9121d6002c0"
RELAY_12203_WAVE27_PROOF = {
    **RELAY_12203_WAVE26_PROOF,
    "previous_request_id": "90740618254a45e9461f5ee948fc9209576ebfefae755c405a094905e26bf503",
    "previous_terminal_file_sha256": "6fd09a8819a1796998b05e14ac1d7a743bad6671fe46dd22e144943f80ee05c7",
    "previous_transport_progress_file_sha256": "c4beb80f40eab4afb15744a8371d14f043d2d2679a4f0233d1534a6db0b0c4ec",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795727440.780899,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE26_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "f1f7f01659bc42fd9b652cb05740bf2b",
        "job_sha256": "73772f5964e83762bfdaa96807d2acbbf2327b9578d05af0d8116b825d8bfe5f",
        "result_sha256": "2b85016e87707a5f65408de237bfdadc108a38264300554c39b5a18cebb8c1fb",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "bfd2c04a4900472388c84af214f5e72d",
        "job_sha256": "7f3c732815fddc7775169e5b19d1f89ccf0cf7cf98ce7e9b4c86693dd17f1fcb",
        "result_sha256": "c6ee950870297765ed537da9fa50582e118eb317378236799cb954d1b8ad62ee",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE26_TERMINALS = (
    "b297d490ae2ecd8e1999e8bd72d63001b33b4bda4516512daf7af12a274c81e9",
    "6fd09a8819a1796998b05e14ac1d7a743bad6671fe46dd22e144943f80ee05c7",
)
RELAY_12203_WAVE26_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "896ae33f5ffde7798213fd4b5b1cc097798388cb2ab8c73bbba7c1e32562ee58",
    "attempt_00.replay_authorization.json": "7ec65254bf7ba81d5c7f426ebaabad70ec272900cd44ea1d41b082c8aba509c0",
    "attempt_01.replay_link.json": "4c1942baf9fd5fd66e57e1f5520ed8348f055a08ef744600dae85dcd52b55cf1",
}
RELAY_12203_WAVE27_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "0f30e10d-3317-47ca-8cc1-7a3c2e8bdd7a",
    "response_id": "resp_022042e8bda5f6db016b089ec8496487d1a5709ffbdcd1a899",
    "conflicting_response_id": "resp_0f30e10d331747ca8cc17a3c2e8bdd7a",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.61737745790742,
    "last_sse_activity_seconds": 196.61737745790742,
    "elapsed_seconds": 207.03115104185417,
    "sse_milestones": {"response.created": {"elapsed_seconds": 26.975886582862586,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 26.977500916924328,
                          "interval_seconds": 0.001614334061741829},
 "response.output_item.added": {"elapsed_seconds": 26.978995874989778,
                                "interval_seconds": 0.0014949580654501915},
 "response.output_item.done": {"elapsed_seconds": 41.70412016683258,
                               "interval_seconds": 14.725124291842803}},
}

RELAY_12203_WAVE28_CONTRACT = "8f38ecdc0aaec090d23eeb9ed52024b6fd2e7dcce325b0d2d88e5ab05b7042e8"
RELAY_12203_WAVE28_AUTH = "e9e083b3eb9c4cfd44f8cb3a35f9d78be8349fc2e95021e9ac0be44ae14f77e3"
RELAY_12203_WAVE28_PROOF = {
    **RELAY_12203_WAVE27_PROOF,
    "previous_request_id": "8dfa3871d1dd0ceb5228520eb0ae6745ac13eb164105f105e0fad3f414013cee",
    "previous_terminal_file_sha256": "ed4e43d6f8253eb7f0211d239fc0e756d13dfb0e85dcefeb3bb14140fcbaa384",
    "previous_transport_progress_file_sha256": "fa0099360fbf50c2c2995aa32a8b1105ab32ba4043a0a9edbe1d048bc320fc6e",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795728573.4999516,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE27_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "a2f3f4d289f348b287cc243e2735d113",
        "job_sha256": "a73080e3e79d533ea8489353b204a779b564a2b57599c935c30b79d24e252e67",
        "result_sha256": "adedafe461242a8a9b47e98ad7fefd85372249f65b6228b844100a7ef7d84584",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "8c5b0a47fdc840b3977f30d25dbcb061",
        "job_sha256": "d07d9a3699207ac0b2402f4ac5cd6c4f8ef2caf84d86cdae59e09ea682c3d3be",
        "result_sha256": "3e5756fa875c45810525fe259aee5499533e4f6c03ceca2ce91992729dd85326",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE27_TERMINALS = (
    "d106da71ec687bc7a581eee9ebe129fdacbe6dd955222639bf7069503f5128f3",
    "ed4e43d6f8253eb7f0211d239fc0e756d13dfb0e85dcefeb3bb14140fcbaa384",
)
RELAY_12203_WAVE27_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "0544c1e458145155eb17a622d88e3422ddddbf73cfc9a728a6f1dcacc2796675",
    "attempt_00.replay_authorization.json": "bde1eaa4e28db61dafd378cea2fb55e1c5d6b46c7c701cb668543d7d177bce30",
    "attempt_01.replay_link.json": "67421cb45910f3bd575661247d05506ace0c36fbf5d3e463179b6b3e40e45672",
}
RELAY_12203_WAVE28_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "1bf68c2c-1f8a-4830-a217-e4d4c1b2415c",
    "response_id": "resp_0b900cbd2f8e0992016b08a335f92c87d0b81b0703126df851",
    "conflicting_response_id": "resp_1bf68c2c1f8a4830a217e4d4c1b2415c",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 200.91070891707204,
    "last_sse_activity_seconds": 200.91070891707204,
    "elapsed_seconds": 208.21483075013384,
    "sse_milestones": {"response.created": {"elapsed_seconds": 28.216866417089477,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 28.218607207993045,
                          "interval_seconds": 0.0017407909035682678},
 "response.output_item.added": {"elapsed_seconds": 28.220885083079338,
                                "interval_seconds": 0.0022778750862926245},
 "response.output_item.done": {"elapsed_seconds": 44.96939670806751,
                               "interval_seconds": 16.74851162498817}},
}

RELAY_12203_WAVE29_CONTRACT = "41f51726abea8f9f41e12c46dd8f3a85b735558b38245cc3d56db971e5408302"
RELAY_12203_WAVE29_AUTH = "17182020d6bcaed7fe7eb1cd074d19ca3ed55766497f4e5b50e2c6260cc4d016"
RELAY_12203_WAVE29_PROOF = {
    **RELAY_12203_WAVE28_PROOF,
    "previous_request_id": "6f298cc3b55eef06e1a679ac588ee7bbb17d8e3e83c453d960f8e84b7bcfdbbc",
    "previous_terminal_file_sha256": "76a55f0d9304bb5465cfb70b50f4a6b22e7e0256533fc6c4290a8235bdef4ae5",
    "previous_transport_progress_file_sha256": "64c94ca0854fde776105d03d778ff07a95121ebb9a005f6cb35bf0aff47cd7c9",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795729614.5429068,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE28_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "28549f09fb7243bc8938f9fa3437e1e4",
        "job_sha256": "2df9575e48f00a43357d7581a5b3c5793f74486cfbeacf313d87ddbdd4ea874b",
        "result_sha256": "0f229a6de810cd0ce1bb57639ed6b6391f6d6c17cc80b037a3f57435e26434a0",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "dd515d6606684aa3bd332499ceaf456e",
        "job_sha256": "245c1679de45a59a6d11c9d75c1582cb516cd8306b7cb08d8c7083e7d1789b8b",
        "result_sha256": "8ba60b7b135cf0033512f918ecdb73436017d16a516e6be9f1b9d052a3c44b2f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE28_TERMINALS = (
    "ab736073f1f8eec5fdde734e2acd9eea470703b75ac25b74ba1b50a1db5c7099",
    "76a55f0d9304bb5465cfb70b50f4a6b22e7e0256533fc6c4290a8235bdef4ae5",
)
RELAY_12203_WAVE28_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "1adc9d264a4db81e2b97624b0338d38414eef25daddd3de8fafc2256913887df",
    "attempt_00.replay_authorization.json": "9d47d05d474e86a53f9e5922a808d2120cdeaa1782342004427b5e1bd5a545ad",
    "attempt_01.replay_link.json": "64397057e0c7f74054abb21dd5421b1d8246a3a8a6375c2391f54dcdd2598990",
}
RELAY_12203_WAVE29_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "c096a548-d766-4bae-8d94-0fb2162161b2",
    "response_id": "resp_0254d74ea878b7aa016b08a74f571487d29731b1d45ba95dee",
    "conflicting_response_id": "resp_c096a548d7664bae8d940fb2162161b2",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 194.4747688327916,
    "last_sse_activity_seconds": 194.4747688327916,
    "elapsed_seconds": 197.2414221658837,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.30073512485251, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.303228582954034,
                          "interval_seconds": 0.0024934581015259027},
 "response.output_item.added": {"elapsed_seconds": 17.305725457845256,
                                "interval_seconds": 0.0024968748912215233},
 "response.output_item.done": {"elapsed_seconds": 26.645467040827498,
                               "interval_seconds": 9.339741582982242}},
}

RELAY_12203_WAVE30_CONTRACT = "1fbfe0e4dccdf7ac06c497ad409a63a61f0521fdc7f0cb1b289e04a2f3b9a5c2"
RELAY_12203_WAVE30_AUTH = "eb1102f3078c8d00d8e6bbe4fbf7b385b183dc8d3b2d5ec6e3cc5160f0834d43"
RELAY_12203_WAVE30_PROOF = {
    **RELAY_12203_WAVE29_PROOF,
    "previous_request_id": "6b7ab741692c51fb04b666d58213b786ae52fa74f722a7fe888886ce6786457d",
    "previous_terminal_file_sha256": "c77bb0c5102f86a3f4efecf0f24e3fa94bf47188d91915108e4483a537343279",
    "previous_transport_progress_file_sha256": "fc62b9466cf6db88e4ea4565b5a10e4212b507963898778e5c6b096e3f519c90",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795730733.2632422,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE29_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "420185dd70764c478a8ca7bd2b202726",
        "job_sha256": "0a3dc3a4accdebc897b717768792b8823e9a3d61563d79ad781e292c016e11ac",
        "result_sha256": "8073f6746f74cc4ca3a1a137f6bbcc0e9072865ef61cba3d0c9a946b71c7b1b4",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "75901ef837ec4280aa8ba975239f2066",
        "job_sha256": "1add75757e5e324368de91dff8344ab19278c56da9c1ef5e3d32c9602e33321b",
        "result_sha256": "de156ca538750c9262129978410a1214ed37a8fa6293761f5f1345b1bf096ecd",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE29_TERMINALS = (
    "a74dda302023589875dbdfa30e8e83aa3d16bd8f02313e21d239fd3f51c45947",
    "c77bb0c5102f86a3f4efecf0f24e3fa94bf47188d91915108e4483a537343279",
)
RELAY_12203_WAVE29_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "28865419dcc582fb1651fa7fd45ce810b2b1114ab2e90eba98f725566862bdc9",
    "attempt_00.replay_authorization.json": "5ba40274fb33f67bea87b04c55133fd89acde42da659b96be8b601194522c28f",
    "attempt_01.replay_link.json": "9d81d29356e39f11896fa03828c86cb4ccc4cdba41ee1166f361b604ef3ff588",
}
RELAY_12203_WAVE30_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "3b8e9b05-66cc-4365-b04e-8f04d0c0ed88",
    "response_id": "resp_0c0db30a37dc2254016b08aba29c3887d1b1bd12e0607cdf6b",
    "conflicting_response_id": "resp_3b8e9b0566cc4365b04e8f04d0c0ed88",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 202.90926029207185,
    "last_sse_activity_seconds": 202.90926029207185,
    "elapsed_seconds": 209.08399891713634,
    "sse_milestones": {"response.created": {"elapsed_seconds": 29.08444350003265, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 29.08614725014195,
                          "interval_seconds": 0.0017037501093000174},
 "response.output_item.added": {"elapsed_seconds": 29.088339167181402,
                                "interval_seconds": 0.0021919170394539833},
 "response.output_item.done": {"elapsed_seconds": 44.03598566702567,
                               "interval_seconds": 14.947646499844268}},
}

RELAY_12203_WAVE31_CONTRACT = "462021cc382b9c5e670ce6f0e9088a89519d4b94b814d832f1ecdc5c54252548"
RELAY_12203_WAVE31_AUTH = "cd3a5b7a2a2fac8b80dc42fc0bde9f5d734f7a7ce34b586505358e87b9da555d"
RELAY_12203_WAVE31_PROOF = {
    **RELAY_12203_WAVE30_PROOF,
    "previous_request_id": "94c24b08cc7645712bb29c71adadd9e56b7e93e72750de563cfaef0759334ccd",
    "previous_terminal_file_sha256": "06a15c892c12fcbc9a2ab12b1aa01864e67f6b63975d5c324959f5da7a7552ba",
    "previous_transport_progress_file_sha256": "20983da09ba62b51866de268630e76ba137575346ec8a1949e9ea202ad4b511d",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795731691.2409678,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE30_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "3eb361a886824203a8eec5c54183b057",
        "job_sha256": "6547c2e41d64aecee67352d34154d5c9c35b4e4ae854f6d6ac32cc01e4fc3a38",
        "result_sha256": "f940c8a64e7f8d11ea656f9cb620e26c28acbf88bdad48a028c68d362f196054",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "b44af33229c94181a6f77130b68a42f0",
        "job_sha256": "28c86e6c492ab6843437c117cad0297342e6da21aee728882b489352926553a5",
        "result_sha256": "64bcf8c80362984f7747bc77e1e4de98307c30d8a92dc5474b89b25507c7abdf",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE30_TERMINALS = (
    "ba8c0852253f1e24e75bf15e08ac64288cf0e6590683836262c28f1674b314c8",
    "06a15c892c12fcbc9a2ab12b1aa01864e67f6b63975d5c324959f5da7a7552ba",
)
RELAY_12203_WAVE30_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "10433bcb30d4235c333f49ca4a1fb395f1b0a1e151d9ec72dd1fce605e098854",
    "attempt_00.replay_authorization.json": "3c13ea9b233790be6d7a3b60a63ef2aed3fd5d4ac8ca7efa41ebd524c2c7fe6e",
    "attempt_01.replay_link.json": "ef5cf69e69a027c9fc39a61369e4230b944becf7f7b7ac30104400e74f5095a1",
}
RELAY_12203_WAVE31_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 43,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b8739ba6-1bf6-4c15-9502-cc58e1eef158",
    "response_id": "resp_0e2e1f934f5fd834016b08af659c5887d0b42ac00ff7cf3474",
    "conflicting_response_id": "resp_b8739ba61bf64c159502cc58e1eef158",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 201.1035465421155,
    "last_sse_activity_seconds": 201.1035465421155,
    "elapsed_seconds": 201.25218083406799,
    "sse_milestones": {"response.created": {"elapsed_seconds": 21.195508459117264,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 21.19700058409944,
                          "interval_seconds": 0.0014921249821782112},
 "response.output_item.added": {"elapsed_seconds": 21.200255792122334,
                                "interval_seconds": 0.003255208022892475},
 "response.output_item.done": {"elapsed_seconds": 30.500380834098905,
                               "interval_seconds": 9.300125041976571}},
}

RELAY_12203_WAVE32_CONTRACT = "6d735a4c4001d1b7d62e8f6d7fbcfee3cda4364704a4502ab1320482c178aefc"
RELAY_12203_WAVE32_AUTH = "4230d790d23728005ccb8260ec4bbfdc78019b778430a75ef46331a2bf42296b"
RELAY_12203_WAVE32_PROOF = {
    **RELAY_12203_WAVE31_PROOF,
    "previous_request_id": "6ad7481a01d7e7b04e4d83b42c5bd83c11d49eb031ae32b8a24ae8765fd93775",
    "previous_terminal_file_sha256": "39a81cc37b7911b7c04436ec5b4317786cf9500b119248c70f49d95b9eb35477",
    "previous_transport_progress_file_sha256": "a25198edfc06d6e4a1d740f57d4b13b8c9301c44c72d7baa5867850615c25c87",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795732690.953343,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE31_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "852f0e2578c94c7a86bd715e199da17e",
        "job_sha256": "da77c3579a256403599059d73ea77f21033f5e7f79a7aa422a893eadc192efdb",
        "result_sha256": "c6a40958c50c318a226d497183c112f9c94e215503489331b70a4508aa6b8ffa",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "4d9dc0c85cd24f71a191f0aaa0c6bacd",
        "job_sha256": "799dd191dca516c78a0d1b43ae3766ac88fdf8d79777c7305f92839d70697fe5",
        "result_sha256": "482f6122ddcac207bfbf3c9af5361cf0acd505de71acc787f4a23e2b3947a5bd",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE31_TERMINALS = (
    "57340e9d29010e856935b860b25261c7f87b15e525ad48dc3b0b765fd9b8637d",
    "39a81cc37b7911b7c04436ec5b4317786cf9500b119248c70f49d95b9eb35477",
)
RELAY_12203_WAVE31_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "c9f18f3e689251d51cc36d4dcf84f020c61ca5fdb9742c1c0d4371a721dcdaa9",
    "attempt_00.replay_authorization.json": "8191e9434037ae76eeb2f8c17a94f933bdbd2c8e5eebf82bf0b1e17006bd9f5e",
    "attempt_01.replay_link.json": "14304edc8dbc3ea44f3135b7a5efc518cae267205d993cd339e878d951d2f462",
}
RELAY_12203_WAVE32_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "a7b9904a-7f58-4c5d-a89a-09da4e90d1b3",
    "response_id": "resp_03f76b932d7a93fb016b08b3538f5487d2b7acaf0b66e0f122",
    "conflicting_response_id": "resp_a7b9904a7f584c5da89a09da4e90d1b3",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.44163441611454,
    "last_sse_activity_seconds": 195.44163441611454,
    "elapsed_seconds": 198.23924437514506,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.324822208145633,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.327101208036765,
                          "interval_seconds": 0.0022789998911321163},
 "response.output_item.added": {"elapsed_seconds": 18.329361208016053,
                                "interval_seconds": 0.002259999979287386},
 "response.output_item.done": {"elapsed_seconds": 27.685390332946554,
                               "interval_seconds": 9.356029124930501}},
}

RELAY_12203_WAVE33_CONTRACT = "a376053a1b6b6e3709ffd761a24d0d897ce9caed65d0b5657eec84a0aadaef93"
RELAY_12203_WAVE33_AUTH = "147f9e6aa37dcd8bd955f228692a4ace6d58fd61dd83964dab28425c37b7727a"
RELAY_12203_WAVE33_PROOF = {
    **RELAY_12203_WAVE32_PROOF,
    "previous_request_id": "ca5a366092db15249283c74226ce3fb64776ffedd83c046dc43247a0700c725a",
    "previous_terminal_file_sha256": "b9c2303c537b73b6bd7c3ba7aacc1b68394059e1070be7c0c6085ec50da6aa57",
    "previous_transport_progress_file_sha256": "2834b61b0db7c4ab402570221969f374498c31d874eca888b65587f72c003766",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795733761.392024,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE32_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "51ccc1d35fb34c3c92688dfc4419ebb0",
        "job_sha256": "fd4e0709ae70f7f2eb51d9a15cd32e990ab3f0c1701b31d047b53fd03095aee7",
        "result_sha256": "a022f32fa50dd402e0fea74b645d5bbde58872142956521a9dbd523f10407072",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "a84c0a6e333241308f1e6b7488d1ce8c",
        "job_sha256": "f6650f6c707b5a2339f5854ad1034ef597fb6c8db90f02a47344bd1275ac70b7",
        "result_sha256": "daefc526dc3dfc30d2fa917d40b546df7b49b20e7d6f964c3d2fdeb6871ef5ba",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE32_TERMINALS = (
    "c143de190893e4a0da5b9ba7440364bd89be5c6afc1e7a6dbd48bfa2dc42975d",
    "b9c2303c537b73b6bd7c3ba7aacc1b68394059e1070be7c0c6085ec50da6aa57",
)
RELAY_12203_WAVE32_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "9e4c28e748d3aa22958f93a7737f8e503cb8ceb5c279cf2b7bdadab96ce6cb15",
    "attempt_00.replay_authorization.json": "57f7a139cd5b0ad37169eb05d18a0bc1d0c2a941d32208d2084a38791c489e51",
    "attempt_01.replay_link.json": "a6321bbbd60ef9164107c129cf0da018827c8afb72f375337c961ad6a86fb127",
}
RELAY_12203_WAVE33_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "531b8a40-dbb7-4a83-a506-fd7655144d8f",
    "response_id": "resp_0f3c1077db2c5b3d016b08b77e16e487d08b2eb88a26d33d9a",
    "conflicting_response_id": "resp_531b8a40dbb74a83a506fd7655144d8f",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.76534929103218,
    "last_sse_activity_seconds": 196.76534929103218,
    "elapsed_seconds": 199.4338793749921,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.460477250162512,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.462459000060335,
                          "interval_seconds": 0.0019817498978227377},
 "response.output_item.added": {"elapsed_seconds": 19.464026832953095,
                                "interval_seconds": 0.0015678328927606344},
 "response.output_item.done": {"elapsed_seconds": 28.807308041024953,
                               "interval_seconds": 9.343281208071858}},
}

RELAY_12203_WAVE34_CONTRACT = "a6b7dd5ef3f3060d5f3ec23792e6ad66b257297fb91999c4ad987a87593943d0"
RELAY_12203_WAVE34_AUTH = "2a43d064dd0809d134e3f4baf55b6f7064bb908e3b89e0b611baaad54f692a5c"
RELAY_12203_WAVE34_PROOF = {
    **RELAY_12203_WAVE33_PROOF,
    "previous_request_id": "edd77546fbc44799fa168710392e212deb05052e4373301bb7c5d721d66bf4a7",
    "previous_terminal_file_sha256": "520997df5973966ac63e1cf9a95417e6117aa1befb46591f769615eb1de8d518",
    "previous_transport_progress_file_sha256": "ae180f69720c0d0dc3713de44ef22a7fc06cb99739f4a3212e85b7d3e5686ce1",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795734737.44935,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE33_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "cac2a722c57549998cc42a219ce95288",
        "job_sha256": "db2e709943f3381ed2328732b84072cc2e2aee3be93765e59d256b7a7b5854aa",
        "result_sha256": "0bebe4fe954385e9d9c40232631f1f78bdd5332c0317e83ae411c85172576ef4",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "3c10b36f4fee4587aeb085be08211093",
        "job_sha256": "29fca5658405e18d35f90cb230a32b32e9c24d45a1d802aa3aab2b1c7745fbc4",
        "result_sha256": "599ddad01cf479b721bd27b194c98d0fca3e362bbf4da5e9eb747dc5b4b6de79",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE33_TERMINALS = (
    "6369b7e46c0eb01cdcf5cb39fbc27a7ac3856505ca1bd199de01deaf37c21249",
    "520997df5973966ac63e1cf9a95417e6117aa1befb46591f769615eb1de8d518",
)
RELAY_12203_WAVE33_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "52ae37aafff46e5ed4fe77dc0a862af53592de9ea6bff1e29f7756a508c4ec13",
    "attempt_00.replay_authorization.json": "369b45158d074dbe8326237e6df9caa0e2542d34816f807cf888b2d38d3903d8",
    "attempt_01.replay_link.json": "8fe4459507e519111f595a934c4c0fc91d8b04742fde4e8d7722b2e311f47700",
}
RELAY_12203_WAVE34_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "827a4edc-c575-449c-92b9-0b50fdd1465d",
    "response_id": "resp_0facc47c298f5f1a016b08bb4dbdd087d1b612c067cf4afec4",
    "conflicting_response_id": "resp_827a4edcc575449c92b90b50fdd1465d",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.6196298331488,
    "last_sse_activity_seconds": 196.6196298331488,
    "elapsed_seconds": 199.24724795809016,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.300910582998767,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.302149458089843,
                          "interval_seconds": 0.0012388750910758972},
 "response.output_item.added": {"elapsed_seconds": 19.30616966704838,
                                "interval_seconds": 0.0040202089585363865},
 "response.output_item.done": {"elapsed_seconds": 28.662907000165433,
                               "interval_seconds": 9.356737333117053}},
}

RELAY_12203_WAVE35_CONTRACT = "5efefa61473344584dfd5dab7384aa58cac3c9ae2af8378e07f15bc6563a5ae5"
RELAY_12203_WAVE35_AUTH = "8cd47b3a796d233ef1168f9b39217c50dcb55baf1558823dba3005d7866b038f"
RELAY_12203_WAVE35_PROOF = {
    **RELAY_12203_WAVE34_PROOF,
    "previous_request_id": "4bfb45f1eccfb73f4c4e267b79a9ceb9b1309301ddf2da27c9e1ca33573db838",
    "previous_terminal_file_sha256": "260d73d6121c87078a82dcc17424bc3b3548f9693b8f4f8f186b32474c61e4ab",
    "previous_transport_progress_file_sha256": "15f87d3aa43f58827ad256a6a923e5b4110848f6566d50a7e5ffe675d2ce84ee",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795735786.489421,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE34_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "881d112f46624bc49634c618350bdd0a",
        "job_sha256": "e36856b9f34b3d3cedd8725c3b0d96bcf724c54ec1b20cdaca046a19649c8772",
        "result_sha256": "8d43241686f62bff70e3ec00f3e65184c6316d94f171d7917fa3ad67d95aea11",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "a081770beebe417eabbe3079894a7991",
        "job_sha256": "6a3c430fd6a4bc2d347f76b6e4a0617af599a056a3aa30a5fd4ba83a1484ccd0",
        "result_sha256": "e5a37e463411c853dd1016120c919e4bb7af001d3d8246aff150ad0a8851d6d2",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE34_TERMINALS = (
    "18abe2c55fb55d79e975f84fbf230f4b7040e2a6f0e188af8fee18e8cd4d33d1",
    "260d73d6121c87078a82dcc17424bc3b3548f9693b8f4f8f186b32474c61e4ab",
)
RELAY_12203_WAVE34_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "9ebcdeca8c355e44c21ab3e2474ac6c9790b02c4d447ce55ee5f7707a1573662",
    "attempt_00.replay_authorization.json": "06b94977ed5d55d16469aefcd0cdc6f9001729607f4d47d88f84e11ecfa8ac17",
    "attempt_01.replay_link.json": "f5c9d76363fafd4b77333855b070b1b0c23c06e3b263a6b2c745083bfd5fed31",
}
RELAY_12203_WAVE35_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "38851e5b-befb-4cb7-afeb-c78d70bb538a",
    "response_id": "resp_0dc77212b8816514016b08bf6769e487d08fd4073677df571b",
    "conflicting_response_id": "resp_38851e5bbefb4cb7afebc78d70bb538a",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 198.3475421660114,
    "last_sse_activity_seconds": 198.3475421660114,
    "elapsed_seconds": 204.19428704096936,
    "sse_milestones": {"response.created": {"elapsed_seconds": 24.219916875008494,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 24.222844708012417,
                          "interval_seconds": 0.002927833003923297},
 "response.output_item.added": {"elapsed_seconds": 24.226195957977325,
                                "interval_seconds": 0.0033512499649077654},
 "response.output_item.done": {"elapsed_seconds": 30.537329416023567,
                               "interval_seconds": 6.311133458046243}},
}

RELAY_12203_WAVE36_CONTRACT = "b5353828adc1af89ee8c58ca4fd60cb7530f8b54924e5e56f6177e86f440388b"
RELAY_12203_WAVE36_AUTH = "d3a7962200cf97894293eb76346bfb7d8f38a1ff67c5ea5159812c0d2cbcdc14"
RELAY_12203_WAVE36_PROOF = {
    **RELAY_12203_WAVE35_PROOF,
    "previous_request_id": "0a2b95b261b3c3ca04ca752a4a2b1501a6f024e330278c67a5d1c5f90cf6466c",
    "previous_terminal_file_sha256": "05684bb58387335cbdb8ea438999ee8e441af3715812353134854524aad93c9c",
    "previous_transport_progress_file_sha256": "4312786e0dfe0d390d6e42340c72d5e7c0cdc1b904de21788ec740b100566128",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795736915.1458318,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE35_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "c6395fd2ee1c4a099302f29cc33336bb",
        "job_sha256": "f00dd8547dd955b1321aae2c066c20957cdfdb771846ec16427f714a49b036ba",
        "result_sha256": "fae5f665c9c64ba06709c48822e829ac1eab6027f6e894f25b8303a10dbc28b1",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "ca104adc27b5428e87a33cce8ea78461",
        "job_sha256": "4c6135adbc254cbd33d8e5bbf5e7d2c0df7b78f38455e9b61b6739da338ce6e2",
        "result_sha256": "08cf141611b6e26dc7466ec6b8f49bf239012ffeff45719aafa17df97b9b714b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE35_TERMINALS = (
    "43da785d03d8c5db5b83530c37616bd5e4b871b6d99e140e026348e738f09710",
    "05684bb58387335cbdb8ea438999ee8e441af3715812353134854524aad93c9c",
)
RELAY_12203_WAVE35_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "081f85924de031fa2a5ad3577ab629291afad77d6dd2ce9053278bd2e0bb12c8",
    "attempt_00.replay_authorization.json": "7c252d834323a312520e163572b47cb7c640c064db5d7b8735b4be0c1f80b186",
    "attempt_01.replay_link.json": "72d12e9164dd005bb14d47b0d700c2432c46580f79217ea15aa2c218b07550a5",
}
RELAY_12203_WAVE36_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "76985478-0ce9-4826-b4d2-17788de14077",
    "response_id": "resp_0f6f63de30d09cf8016b08c3cf3e5487d1a72b29cf32262b47",
    "conflicting_response_id": "resp_769854780ce94826b4d217788de14077",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.66865158313885,
    "last_sse_activity_seconds": 195.66865158313885,
    "elapsed_seconds": 198.44547879206948,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.4289267079439, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.431571166962385,
                          "interval_seconds": 0.002644459018483758},
 "response.output_item.added": {"elapsed_seconds": 18.434350166935474,
                                "interval_seconds": 0.002778999973088503},
 "response.output_item.done": {"elapsed_seconds": 27.692825792124495,
                               "interval_seconds": 9.258475625189021}},
}

RELAY_12203_WAVE37_CONTRACT = "b0e28b7a5db81550512f3c905066aa0a0f0de63b0c8ed5684dc3ba36c53382ff"
RELAY_12203_WAVE37_AUTH = "17370f25645a0f1acf7d2afef89eda30383a37f2315d5ab871baf6ee7ab00a39"
RELAY_12203_WAVE37_PROOF = {
    **RELAY_12203_WAVE36_PROOF,
    "previous_request_id": "f8cc016478ebfb78bb34e3635a6edc46d00587a8eb6eee979e14a5e9fde4665f",
    "previous_terminal_file_sha256": "1e03f1f66ef520b70c110e3bebde63c3bbbb569420f2e1a7089f720427e1092a",
    "previous_transport_progress_file_sha256": "23e15fe7c1e65e511e9e549f1f928cb9f028ccad0315a3cfb7d0a367d8736eea",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795738859.9096358,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182},
}
RELAY_12203_WAVE36_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "8dd2bf5e7b534fe98d0e503bbe3a6aa9",
        "job_sha256": "78758c0a2c3c40631ef93be7c199683ca057d8e599e87c2e73208df759f8189e",
        "result_sha256": "644970d6fdf52105cf313b3532a0977f77bc4a3a162b7f6540cf9fb00c56014a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "34d242e3acfd409694cb8b2ea726207b",
        "job_sha256": "8692da43d687a99e8bad6108e30f28f252bfd9bbdd02c49b554494d394a81ab3",
        "result_sha256": "1e8ea399f81e5456c49f35d6d6a14953dfe4b525cee92e9a3618b95fb9fa12be",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_12203_WAVE36_TERMINALS = (
    "38d916218e10819c6c3d04103670866ababf2cf9c13c9830decdd27e152b6521",
    "1e03f1f66ef520b70c110e3bebde63c3bbbb569420f2e1a7089f720427e1092a",
)
RELAY_12203_WAVE36_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "96b8198c4fb6a6c3b16df7c80ae50724648d3cabbc21e9849e8c5f25bdd8f31a",
    "attempt_00.replay_authorization.json": "de21b70810d64755a696a1db00294e6190910be4c14eb0420eafb29420c119f4",
    "attempt_01.replay_link.json": "13392f95873920009a7cd026e5a4df697a3bb6f28dbb2ee45e07c211d046b03b",
}
RELAY_12203_WAVE37_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "11273ba7-bbd6-4bd4-9d22-fce3d500f818",
    "response_id": "resp_07ce62538d2a65fe016b08cb6e0ad887d09be558084d811be9",
    "conflicting_response_id": "resp_11273ba7bbd64bd49d22fce3d500f818",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 199.91539245820604,
    "last_sse_activity_seconds": 199.91539245820604,
    "elapsed_seconds": 202.73213687515818,
    "sse_milestones": {"response.created": {"elapsed_seconds": 22.75132487504743, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 22.753435416147113,
                          "interval_seconds": 0.0021105410996824503},
 "response.output_item.added": {"elapsed_seconds": 22.75551191600971,
                                "interval_seconds": 0.0020764998625963926},
 "response.output_item.done": {"elapsed_seconds": 32.06008116598241,
                               "interval_seconds": 9.304569249972701}},
}
RELAY_12203_WAVE37_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE36_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE35_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE34_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE33_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE32_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE31_ARCHIVED_SSE_COUNTS = (41, 43)
RELAY_12203_WAVE30_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_12203_WAVE29_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_12203_WAVE28_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_12203_WAVE27_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_12203_WAVE26_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_12203_WAVE25_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_12203_WAVE24_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE23_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_12203_WAVE22_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_12203_WAVE21_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_12203_WAVE20_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_12203_WAVE19_ARCHIVED_SSE_COUNTS = (41, 40)
RELAY_12203_WAVE19_SOURCES = {
    **RELAY_12203_WAVE18_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "d7eceffe7ebb9bc5816e85f931df0e0ead2a1698035add79437f8ef96df97448",
}

RELAY_12203_WAVE20_SOURCES = {
    **RELAY_12203_WAVE19_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "51b30b6e1b75163a4bb6916e1d1eae114cf257287dd05b50a1fe9b4bd967a604",
}

RELAY_12203_WAVE21_SOURCES = {
    **RELAY_12203_WAVE20_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7169e6a2b776f13f80b85cd603fa6be7846c35dd3651791d1be73465de5c8577",
}

RELAY_12203_WAVE22_SOURCES = {
    **RELAY_12203_WAVE21_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "91ace6e7dbc2c80472a70fbb86ad772630dd87c7d2ffa85df57ee04a55b8b011",
}

RELAY_12203_WAVE23_SOURCES = {
    **RELAY_12203_WAVE22_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "06f5b65c4486d85aa86abd4d5d36f1813f0b5e7e2982c09a8ee08a717a733916",
}

RELAY_12203_WAVE24_SOURCES = {
    **RELAY_12203_WAVE23_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "4f8b21df0a87977169efdd448caf903159a507eec0803742c546bd57ae559831",
}

RELAY_12203_WAVE25_SOURCES = {
    **RELAY_12203_WAVE24_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "74750aefc2e457cf5a946fe35ca94f88154f6674eaa7b1c253905e729b038a02",
}

RELAY_12203_WAVE26_SOURCES = {
    **RELAY_12203_WAVE25_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py":
        "5ea4b2415fc2779471990e5192d9ef9a3be1f5233b20b42f651c034f6d493774",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "e10ce54e7b0ba1147520d6f36cf89ef541984b9a64ffa83c03b0ba26582f392c",
}

RELAY_12203_WAVE27_SOURCES = {
    **RELAY_12203_WAVE26_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "32379dd1f42b4fb431652f18e8ba863b3f29f61fdea74afa08e2124c9a80ebac",
}

RELAY_12203_WAVE28_SOURCES = {
    **RELAY_12203_WAVE27_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "80f828e8ee46edda330b64ffaa50779d2fddbfac3c3aa72011d9c17655fbf116",
}

RELAY_12203_WAVE29_SOURCES = {
    **RELAY_12203_WAVE28_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "1cede93b27c83c702b1acbcbcfda6488cc67af43c089ed79926c0eebb93ff6b3",
}

RELAY_12203_WAVE30_SOURCES = {
    **RELAY_12203_WAVE29_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "5bd2b0eded7156420ce9f9a685247ad008e2b316c681d1aa03cdaa2ed48f50cc",
}

RELAY_12203_WAVE31_SOURCES = {
    **RELAY_12203_WAVE30_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "4899aa139ced91cc84f5a42e64a011b157775eb77ec17e330637445241f8ee72",
}

RELAY_12203_WAVE32_SOURCES = {
    **RELAY_12203_WAVE31_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "6c5dacfed84eaedabd8b6684fb46b24b25163bb82eb2a0461913b58c67dda3f2",
}

RELAY_12203_WAVE33_SOURCES = {
    **RELAY_12203_WAVE32_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "ffb802c433ea5a3891463544645a581907475495033579d064d563a0e8cb6aa5",
}

RELAY_12203_WAVE34_SOURCES = {
    **RELAY_12203_WAVE33_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "f30d389118031545d03597fc3a107cc9b8b67d7474b59e86c2cb679a40c275e0",
}

RELAY_12203_WAVE35_SOURCES = {
    **RELAY_12203_WAVE34_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "23eb4b7e9415ea3053d67646ba8040a91b732853b620928a687a0568eddf3f4d",
}

RELAY_12203_WAVE36_SOURCES = {
    **RELAY_12203_WAVE35_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "586fe2cf2e30b48c111aacfccd0f241fde96f2e860d497b0dfda1b76c582b5c5",
}

RELAY_12203_WAVE37_SOURCES = {
    **RELAY_12203_WAVE36_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "15c2577015fb312eab6038399f2ae2d5f894407923d884a8a7d6673fe17f20fe",
}
RELAY_6101_WAVE09_CONTRACT = "c5a2f7184c17057fe9b9691dd3ba9ad5edef8a7c4d912adcc1dc9056927ba37f"
RELAY_6101_WAVE09_AUTH = "12285ae28447c96809134b8c38b20512ef5ab5040f36416fd15ff67ae8ccde1a"
RELAY_6101_WAVE09_SOURCES = {
    **RELAY_12203_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "799e7196139a18c28f51d308ed2edd01bf3d935d47bb68cde4bb83d22f4b3485",
}
RELAY_6101_WAVE09_CHECKPOINT = "5f2b2b644f40d3e2dbc91ca5da7bab9003eadbeb37623ee9bd0c2d777d1f1a9a"
RELAY_6101_WAVE09_PROOF = {
    "logical_index": 0,
    "payload_sha256": "a59f16ba3c689a594bcbc566b972da3f22a6d00adb1483dad0290bc75dc4cb9f",
    "previous_request_id": "e4920fc471d314f94e30ddd5cae8296b2cc00c61b644fd0befad4f796e8923e2",
    "previous_terminal_file_sha256": "714cd62cda2b9507ebfa0a200f0d60304cce0aa9d48f3d5120f448746c20c7dd",
    "previous_transport_progress_file_sha256": "8ab403e2d74c70a9d8d2f806f71b4c0a9643c7e7bc68b765be7aa50dcd84b0aa",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795710450.8833244,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 30, "maximum_logical_requests": 88, "maximum_physical_attempts": 175},
}
RELAY_6101_WAVE09_PRIOR_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "69cc4c61d3c841528eac07c9648f93ea",
    "job_sha256": "91183e20f85e4502336ee692afdb57a1dc44b7084cf85304aab63a35b60ddef9",
    "result_sha256": "3cf63864aed8906b58a961d818f7de0fa0026085759ef5abcfd751873e320a1b",
    "error": "connection_interrupted",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_6101_WAVE09_FITTED_FILES = {
    "boundary.json": "d9315cd18eff5ae0ec26414f398d210929caabfb02c4fc3a3cff5b9346e6cdff",
    "round_05/source.json": "ff67b9cd13896991283e9491fd5856b0fe88ae31cff4067d40da46b4503bfb94",
    "round_05/boundary_memory.json": "3bfe5d7f06bd1c4162b6ef14e307a186b9d2c3775d600622bb42fd731911fbce",
    "round_05/selection.json": "4d40f61edac613254719cfb3a8e2a06d5123d8c90bd6837fab8fea39e0ed2316",
    "round_05/batch_freeze.json": "6dbade356cac58ef256082277d41241afe8d1433b9cfc07e2f1dfc95c0bd9c7e",
    "round_05/memory.json": "7b6180f12da4b0d87e428f8e7b7a42ddd656a34115c69f6a4b4646fc52062997",
    "round_05/turn_00/attempt_00.terminal.json": "bacecbda3c91638c2c07fb52c0dcc490fedfcab27bfab9d2c9b13f62ef0cf273",
    "round_06/source.json": "896eb6ebbb89b83c6cfd6d34e4af8d322d5ee593e7dfc0c73a9277f93283219e",
    "round_06/transition.json": "97315c1d7c2f92b940d3ec9e30841b4496eec6dcd5f04a84b9ab832b72602e18",
    "round_06/fit_000/fit_audit.json": "4d03eece5896e116fc35aee450309264bb87a96925f37bdd71772da538752df6",
    "round_06/fit_000/validation_metrics.json": "55e2385893ab29c2f0d5b8d0d49b37cff5aa0c5861230a6f53deee377cec6cd5",
    "round_06/authoritative_packet.json": "49d34f1bda4e39ea826fb37be1c094a8072c2be63b40f17b9128e44ad4a5183c",
}
RELAY_6101_WAVE09_OBSERVATION = {
    "phase": "transport_failure",
    "error_type": "ReadError",
    "elapsed_seconds": 22.604738916968927,
    "network_phases": {
        "direct_ipv4_preflight": {"duration_seconds": 1.0998312919400632, "observed_seconds": 1.1062773340381682, "state": "complete"},
        "direct_ipv4_preflight_1": {"duration_seconds": 1.0998312919400632, "observed_seconds": 1.1040142921265215, "state": "complete"},
        "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.3846792501863092, "observed_seconds": 0.3892637090757489, "state": "complete"},
        "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.3268952078651637, "observed_seconds": 0.7169538750313222, "state": "complete"},
        "receive_headers": {"duration_seconds": 16.4222404579632, "observed_seconds": 22.599786292063072, "state": "failed"},
        "send_body": {"duration_seconds": 2.0461973340716213, "observed_seconds": 6.1744883339852095, "state": "complete"},
        "send_headers": {"duration_seconds": 0.00022587482817471027, "observed_seconds": 4.127967624925077, "state": "complete"},
    },
}
RELAY_6101_WAVE10_CONTRACT = "fb0342ccd7700c266ee0a460096aaec248884996955edc20a9af8e30fe2e2137"
RELAY_6101_WAVE10_AUTH = "c1ab36d0ff790098b4782cd9438e893aa060bd8e980cb4b3fb6242086c801ea7"
RELAY_6101_WAVE10_CHECKPOINT = "244c8df2fc40b2b83f0abdf5f710e419d15ebd91918b7bad2c1c33a48d45da8c"
RELAY_6101_WAVE10_SOURCES = {
    **RELAY_6101_WAVE09_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "4cc7162c414f88f71804a0c6f4abb93091928431b504d5ea6d73183fb65e2e40",
}
RELAY_6101_WAVE10_PROOF = {
    "logical_index": 0,
    "payload_sha256": "2958938fcb0b6cea2e5ef5fb8b379cc02a4ed8a35b4d5635a2fd7f3eb95a006c",
    "previous_request_id": "f357136bf6563f4c50d12932d5ee638c90ea9f33cddd1cabf5437694dcc975f2",
    "previous_terminal_file_sha256": "12ef48c3857621ffb6098984e82adbd4345a032c2fb095b1631213af8babb6d1",
    "previous_transport_progress_file_sha256": "bf63990032e2cd82564f0da6cf99f7bf8519251e25f5869e579f2effe8de4527",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795713600.7855213,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE10_FITTED_FILES = {
    "boundary.json": "12df247e9e7c256220eb13cf455304ebd63f6ac92563d6dd7eb29ca352d5052c",
    "round_06/source.json": "896eb6ebbb89b83c6cfd6d34e4af8d322d5ee593e7dfc0c73a9277f93283219e",
    "round_06/boundary_memory.json": "7b6180f12da4b0d87e428f8e7b7a42ddd656a34115c69f6a4b4646fc52062997",
    "round_06/boundary_user_message.json": "a62e1542e0d54e567ee19c270aed97db9d5a6620421c846abf16ba32b6ddf83a",
    "round_06/selection.json": "fcfb53630734b4d6c3b9ee60f25f9dd08581ffdb1a38dea1b3d75d8e96b7caa5",
    "round_06/batch_freeze.json": "dfd739851d29a43be92ef58dff8d651ccaa8aca937e1584d7d32b002549313be",
    "round_06/memory.json": "e5201f76f9d49c89ae278d6d2e82ef58cecdc120d145c6037364a2536595da20",
    "round_06/turn_00/attempt_00.terminal.json": "97a5a6e8648b1e9bd58aa7f9251b63ad5d1519e929ae070dd80122f8141f4216",
    "round_07/source.json": "74098c938e8ed4e1ff3f09af639b3bf668bbb2ad6514ddc04a40ddec1046578b",
    "round_07/transition.json": "56631c6ef5a417125f6def6f8a4492ec3dd4897b15d1c7e381b57ee66fe772a5",
    "round_07/fit_000/fit_audit.json": "b6b5a085b8b902276e079d1169c4c5824c8c32557b49eeb5261e927aadeabd9a",
    "round_07/fit_000/validation_metrics.json": "52d462ac148ce7dc8f9926f696e0473b7fd8147f2618ebe3ea2c046db0a2d385",
    "round_07/selection.json": "6e8d8121c2439198a7fa0c62f1ea588451b6adea3befdbd786562dff459aff72",
    "round_07/batch_freeze.json": "356d5a9e0851fe4a77c9df4f26e667121ac6e9639c55ba4469bc7c2d04e7cdad",
    "round_07/memory.json": "926c576d6a1694029b1dc34745d7fbde45934a9d3b87078d19e714dfe62a660c",
    "round_07/turn_00/attempt_00.terminal.json": "27ca3270bdf039c451d79e0930f225abdd434027fad0700fdcff9a3902b92f1d",
    "round_08/source.json": "3230dd9d4a8cf6c0c1222ef322f5240b679d4d7ad17bfde42b16b3134a67270d",
    "round_08/transition.json": "b857ed3cb6c60ad991758756502374a320e5e67b19b07ce31a636c69b57b99c8",
    "round_08/fit_000/fit_audit.json": "1a56dba2787c88d2343742556c375bebcfa6172d41d338c42f2eab2548576cab",
    "round_08/fit_000/validation_metrics.json": "ffb5d6aedb73511ae3146a61a5c7f9725d9957b287651f9b977dca5625ba6364",
    "round_08/authoritative_packet.json": "6b20eb5704f7071f78e9ffb057c8e0bc9f38d1ae9c95d31e187a1fa63ea0334a",
}
RELAY_6101_WAVE09_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "ef5051cc66524f29966df442211f160b",
        "job_sha256": "4867d3f8bfae897d5d11943b9d440474dedfa4ebc8ef89be17a914b9caab0ea7",
        "result_sha256": "5e3d759fe78b7d471b6f0e60bf5748f88a712c1af25b62ebe854853635b0d5e4",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "450d602d7de84cbd937750f11456c549",
        "job_sha256": "088d082ca7ad73b25523ebf3b2b05c8d4283eae651b078dc3bc0eb0d28f8bc95",
        "result_sha256": "8b895d8a1fb1aa8a883b7c8a55b60ac9548bd8ecd2844a297105dd4d16aea7ae",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE09_TERMINALS = (
    "48cde2fd1f636e008383f035e88d98f3b4d888ac61446e5992d8d42f71802685",
    "12ef48c3857621ffb6098984e82adbd4345a032c2fb095b1631213af8babb6d1",
)
RELAY_6101_WAVE09_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "cb6f2bb92c02f76c645a66e9441799f025309d53400fd81098adb0f6942dd829",
    "attempt_00.replay.terminal.json": "a5a190fa2f36d05fac6960a9834dff11859cfb8aef955684444ff9a698a8286c",
    "attempt_01.replay_link.json": "7a082d4289a51447d8ea47391fbcff458799d5ef43f6076a5be1f2fd9abd6467",
}
RELAY_6101_WAVE10_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "cf4c6250-dcff-4f63-a2c0-5ed6e3d40527",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0424cc53454e7b6e016b0868b5a8a487d192173d72e36d844a",
    "conflicting_response_id": "resp_cf4c6250dcff4f63a2c05ed6e3d40527",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}
RELAY_6101_WAVE10_ARCHIVED_SSE_COUNTS = (24, 25)
RELAY_6101_WAVE11_CONTRACT = "2eeb40775aad7cb3364c88bb61fda34302649421818f96d993496a4b5ef7bb37"
RELAY_6101_WAVE11_AUTH = "1208097b53d07ee3ef6bc753abf7b839a717e3e2eb11f23e97ef2ccfc028f2b1"
RELAY_6101_WAVE11_PROOF = {
    **RELAY_6101_WAVE10_PROOF,
    "previous_request_id": "5b8ac84b3646090579664633a02389af05be6e42684cea78ddaf25fcf540c0d4",
    "previous_terminal_file_sha256": "9bc303dad617a073f4d5876441d8e556e2884c4649d9d09ce630bdfb0dc755f6",
    "previous_transport_progress_file_sha256": "cbd90d57c55e7561775e41c8fdfcd6cd6601b5542d9be84352bcc0854af69966",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795715004.6630635,
}
RELAY_6101_WAVE10_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "db13d3243ddc4591acea7c94d2fe4b75",
        "job_sha256": "c8fe143b6d00331d23ffcdf2e8e66194887e4dddfc00dcfcf0171bf75c56b3d6",
        "result_sha256": "8e2719bc25c2f5ebdbb1b165301bdc48133a104ae62062d463a08c0e37ca9364",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "55a4b2e1c9294a7184ca133aecca3577",
        "job_sha256": "0a9afe0b83dfdccf855f6e57c5341732e007930a04def44c1c2c103da3121b48",
        "result_sha256": "2a446ca7c25cf49d7a40ab02c95e84fa78ee14067fefeee2d43496cda9c1100c",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE10_TERMINALS = (
    "6e31bafc4dc07f499a8c8f63ff4b96d27a5418357dc128d169d1eecaae7e42eb",
    "9bc303dad617a073f4d5876441d8e556e2884c4649d9d09ce630bdfb0dc755f6",
)
RELAY_6101_WAVE10_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "07c005d90d5a536a1fc547d8f822989750272ad93678863933e96ccef102fc9b",
    "attempt_00.replay_authorization.json": "d4aa7af3b434358d8ba854c8425f61b930e4ae9fecf5d2c5447a7306f5e76b67",
    "attempt_01.replay_link.json": "aa1a28b08103f5353c6dfc0b0a256bf4079b917b693ca80ae74a9f40032f4436",
}
RELAY_6101_WAVE11_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "provider_request_id": "52fa244b-8600-49f8-8101-f301db1daddc",
    "model": "gpt-6.2-astra",
    "response_id": "resp_02114b47e9dc9ab9016b086e2931e487d199dc03f20afa22e1",
    "conflicting_response_id": "resp_52fa244b860049f88101f301db1daddc",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_6101_WAVE12_CONTRACT = "ec792611cee180e42dec88cb93e2285d5adbf44ef35070889b90553ecb42856f"
RELAY_6101_WAVE12_AUTH = "3383ca07d78e1d3d53cfa6e68a0fc5fb74a2309d61470ff931b329272bce0b7a"
RELAY_6101_WAVE12_PROOF = {
    **RELAY_6101_WAVE11_PROOF,
    "previous_request_id": "b20fb55f997040325e772db5ec201fb544bf865c80cf4caf348899f1537e08b0",
    "previous_terminal_file_sha256": "5ad5c1f04b9ed476f609a395a5f725c1846d2c522586ff83df5ca927e1c9dee7",
    "previous_transport_progress_file_sha256": "00de6efcaf267be3986f35d013b6f9b3f22f6ebdb056fb97d356e3bc75c2b2a7",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795717172.261229,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 163},
}
RELAY_6101_WAVE11_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "5b0bd7c2d18c4d19969336f0e21bc48b",
    "job_sha256": "81870dcb524a73897baed11656fa066e1d18784c44db962f6ad083f278353db2",
    "result_sha256": "1c60263f7204b8d55d4ce4361ff1170d029abc84f6996c5983e2ac168d63661d",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_6101_WAVE12_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.added",
    "sse_event_count": 11,
    "last_sse_seconds": 98.36479816609062,
    "last_sse_activity_seconds": 98.36479816609062,
    "elapsed_seconds": 998.379541415954,
    "provider_request_id": "92245a07-bfb7-49d3-b38a-6de5dac41c94",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0ed0dda73e39e4c1016b0871d5f55087d1a9cbbb40341c4100",
}

RELAY_6101_WAVE13_CONTRACT = "eaf4be1327059e4428c44f2b18dbead1bc6939348ff4a42a1a38db63d5c1d58d"
RELAY_6101_WAVE13_AUTH = "bb4a095411844fbb77264300f8768648919bc837cbfa22e806a2f1eb7639702f"
RELAY_6101_WAVE13_PROOF = {
    **RELAY_6101_WAVE12_PROOF,
    "previous_request_id": "76165309548af59ee119b62329afae6bf21af11a7605a89651565026ae585068",
    "previous_terminal_file_sha256": "c533a79262db1228331fdd145f3c474f66e504d48cafe49ae57f3c1e5e70d929",
    "previous_transport_progress_file_sha256": "b77f4fe150f0b5e120c97ad47cb231ab992bb47fe6cc884d737ebaed889c19f8",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795718134.8001642,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE12_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "4931381e0e68465ba95d5bde6d7250b5",
    "job_sha256": "6a41520e25cea9c87772f7ffd72044d92f835d1339e2a6c0887538f4b4831a32",
    "result_sha256": "60217d22b4a367ab3a0115c136a29a90b6d7f75d0b767bc098a11f791b2d1122",
    "error": "response_id_mismatch",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_6101_WAVE13_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "provider_request_id": "dad38e94-87e4-4511-b337-8fd9053b5ab0",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0c1d1ca729a0b81e016b087a78ab2c87d290fba89d87269c80",
    "conflicting_response_id": "resp_dad38e9487e44511b3378fd9053b5ab0",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_6101_WAVE14_CONTRACT = "eb4f325faeb5ba5dd07d2af86521c2c87834b2f7284ed10eb398cba0ffbfaad7"
RELAY_6101_WAVE14_AUTH = "7ea360bc7bdbe2554e698f4f572395bd7c90acfdd038b97ef34ba9bc04fab3fe"
RELAY_6101_WAVE14_PROOF = {
    **RELAY_6101_WAVE13_PROOF,
    "previous_request_id": "2e8bb0c438ec58d8835e619404593f50eab00f6ddf6bb5ac968ae69330a6d56f",
    "previous_terminal_file_sha256": "1f9916f971363cc387bbbe876cfc808b8392dfc1c43c3b589c12c5fbe81b7ab5",
    "previous_transport_progress_file_sha256": "0f88e98983372369406a082573daf3410a7c25c774c40a370337170d5fb58f60",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795719109.4587774,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE13_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "5838b6000a814025b2d1d852f34fdbec",
        "job_sha256": "0f2e69d2f255c9c5049b62eedf28dfe04b37989fe4f13bb808b6d2e03ce1a4ae",
        "result_sha256": "f182276436b741654d035b46600990bb74070eb7581530bf322e5d12b8ccadbc",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "3c292dd317494e3c819dfd4d2d91e11f",
        "job_sha256": "89f7c6019a51fb94bccdc21ddfbe6836718a6c401b5d851196039c8d7b638b55",
        "result_sha256": "0f65080f38dc543b90be42086428d6f5156ece7bc288c979fd537257151ca73f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE13_TERMINALS = (
    "643680653c93985cc863f6449c531baaf78c8a4dbcfd336b1af681c4e34f4310",
    "1f9916f971363cc387bbbe876cfc808b8392dfc1c43c3b589c12c5fbe81b7ab5",
)
RELAY_6101_WAVE13_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "754e073857d2b1b6ef25250256ec83dc2a20eadd459cfd26d28e213e685f1a85",
    "attempt_00.replay_authorization.json": "b3c07aa957c95612cefcec3a0402b516127d8d36075b2f487b5288fa0e95163d",
    "attempt_01.replay_link.json": "1047670bc75a59f934af03d6654074437de21d2c8c4d8faaa9c3288a92640af0",
}
RELAY_6101_WAVE14_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "0234a87c-69be-42c9-9892-09978844c446",
    "response_id": "resp_0ea0a1e91e9a3f39016b087e445cdc87d285e631ffd5c54de8",
    "conflicting_response_id": "resp_0234a87c69be42c9989209978844c446",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.1219058749266,
    "elapsed_seconds": 199.487576500047,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.544332250021398,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.54631866607815,
                          "interval_seconds": 0.001986416056752205},
 "response.output_item.added": {"elapsed_seconds": 19.548579999944195,
                                "interval_seconds": 0.002261333866044879},
 "response.output_item.done": {"elapsed_seconds": 28.862903374945745,
                               "interval_seconds": 9.31432337500155}},
}

RELAY_6101_WAVE15_CONTRACT = "723888e402d1a35102e85343c6d467008d95b4780d334efb8adb8828d55abe04"
RELAY_6101_WAVE15_AUTH = "1f223585dcc62458f3f0ca03bd4b17fbd19bddc84186a814923286c7daa1e956"
RELAY_6101_WAVE15_PROOF = {
    **RELAY_6101_WAVE14_PROOF,
    "previous_request_id": "152b0e1afcee48181046a4d56dc9549d0311eb91a847967f808e439375b1ba59",
    "previous_terminal_file_sha256": "e575b6d466dd59367264e0199e0604f317fdc3d70434714517deec1048601994",
    "previous_transport_progress_file_sha256": "58b78f27278f6d1d9dd56446b70d95aff13fea57a7221a7a20d991bc50ca14f3",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795721303.602687,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE14_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "28a3cea9b05d46acb43d47c85fb1f019",
        "job_sha256": "c6fe885fcf54eb804e419a28c7d77500036e0f0be1391377a6645c7d695f6233",
        "result_sha256": "130e22f72fa4f09382241283192a3dda46a37832efae218b92f6f76c590a267f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "88e82c797f1647ef9bc5b30796f4ce44",
        "job_sha256": "ffa3ace4ffa5ac6914c767a154aed4ba0ec2cd0ad3c3c12fb48360d5cd22adfb",
        "result_sha256": "23a76da405428d3ddbe046b01b96f7f2201a4249fce7cf19050f47001210dffb",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE14_TERMINALS = (
    "d06cba12bacec23e9cb9119101094016068b0c917f3214163fe5f3b43ce4222d",
    "e575b6d466dd59367264e0199e0604f317fdc3d70434714517deec1048601994",
)
RELAY_6101_WAVE14_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "29446f425f5a360dac2c5e56274367ce5af0deb1200829e1c22c7a43e693d60b",
    "attempt_00.replay_authorization.json": "44751e2c62e1936d9b68cb9843c6f5d43fb63b6a86a130006e41fd7d48100d1a",
    "attempt_01.replay_link.json": "0ceab90e3e9c14e153ccd97cc87cb3a941b93ea62cf95864fdd45b8d641f3ade",
}
RELAY_6101_WAVE15_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "33e82de3-9228-42ff-a99d-29ab758c79a3",
    "response_id": "resp_0dcd77ba68925457016b0886d00d6487d2bb1b2ae88af05f16",
    "conflicting_response_id": "resp_33e82de3922842ffa99d29ab758c79a3",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 200.20737820793875,
    "elapsed_seconds": 202.82111550006084,
    "sse_milestones": {"response.created": {"elapsed_seconds": 22.877719667041674,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 22.880311083048582,
                          "interval_seconds": 0.0025914160069078207},
 "response.output_item.added": {"elapsed_seconds": 22.882771166972816,
                                "interval_seconds": 0.0024600839242339134},
 "response.output_item.done": {"elapsed_seconds": 32.2034473749809,
                               "interval_seconds": 9.32067620800808}},
}

RELAY_6101_WAVE16_CONTRACT = "2089fb222a06744f47099a0cfceeb0afaa5504af7abf7e4d8a3393ea92c4351c"
RELAY_6101_WAVE16_AUTH = "01b5321fb259735fc5472494861c66d0a893eb4f66717d9817231eea8a033eed"
RELAY_6101_WAVE16_PROOF = {
    **RELAY_6101_WAVE15_PROOF,
    "previous_request_id": "02285fe9d216e386e4920b659665ca66b29c2ca43ef0f7eb43ec319538f3fbb2",
    "previous_terminal_file_sha256": "17ab86b5f20098b8227442fe27eff12973bddaee3a34f13511b8d3f269a520bf",
    "previous_transport_progress_file_sha256": "338a5c46c9ca48dd39ee9cb87b02a8ba6e2fb46dbc5209ea034a46137b592238",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795722288.4976568,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE15_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "39b38386aaf54b998a6c652f004dbbdc",
        "job_sha256": "b8f5ad7f5306595851bcf713b14aef5e88650c2a9521aa8044634a473f60fea9",
        "result_sha256": "c41d84efad4bcfb88bc55c386aba085bbf33e1c2515f128aa38bcc670df0bb72",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "c9dea397edc7441490ab906504560983",
        "job_sha256": "5040f87fcf2b84bee9bb2da33aaefcf84ca1bbd7c6eeeb4c7eea2a37193588d0",
        "result_sha256": "1abeb6c5d35975c07021d81f6e02759e9cef7c2694f605b36b0560911d7c0a6b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE15_TERMINALS = (
    "aeb856024cab2c5e5a79e17973fdf34ae8186d63705558c238fc32046e8bed45",
    "17ab86b5f20098b8227442fe27eff12973bddaee3a34f13511b8d3f269a520bf",
)
RELAY_6101_WAVE15_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "cfa77fa69645205519bca8ac56852406b059486defcde7a5922e97857469dbb9",
    "attempt_00.replay_authorization.json": "5a0c729d6f585c09bc54a695b2ee424e55103ef82201d0295b8ae2ead113fe73",
    "attempt_01.replay_link.json": "46c395a008f59306e1d106630a061c9e44882f898605d99df303e4880d9d6cce",
}
RELAY_6101_WAVE16_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "284acfc7-42ce-492e-91be-22250147e1e1",
    "response_id": "resp_0026a862684924ce016b088a9d52e887d19b47da308bb19652",
    "conflicting_response_id": "resp_284acfc742ce492e91be22250147e1e1",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 213.05853545898572,
    "elapsed_seconds": 213.06899487483315,
    "sse_milestones": {"response.created": {"elapsed_seconds": 33.16971124988049, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 33.17170966695994,
                          "interval_seconds": 0.0019984170794487},
 "response.output_item.added": {"elapsed_seconds": 33.17382129188627,
                                "interval_seconds": 0.002111624926328659},
 "response.output_item.done": {"elapsed_seconds": 50.751691666897386,
                               "interval_seconds": 17.577870375011116}},
}

RELAY_6101_WAVE17_CONTRACT = "d9eb108d8544c497d71e69804b1082a90fd6667f15bbbabc35863d0114b2d63b"
RELAY_6101_WAVE17_AUTH = "55164af8c4bd97ef8ccbe708254fde01b0464d18b6cd6a1c856b80f219ff17ab"
RELAY_6101_WAVE17_PROOF = {
    **RELAY_6101_WAVE16_PROOF,
    "previous_request_id": "d3d16c7894ab2873d14938dcfed40cd2d9de7685302b9726890a081fa6120a06",
    "previous_terminal_file_sha256": "f37adbe6ee937f470dc7b0ea450be5d8922d3a34688134c6612953681786e1f6",
    "previous_transport_progress_file_sha256": "27ca1b503cc7383c747ceef031d0ed2956c5dcf66850f922ea05f0e917361146",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795723499.4104714,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE16_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "f63231e3fdbe466b9672a8d670113637",
        "job_sha256": "139fed5fb4e868aad97bc2e22d48b2e4bf52c5f6ec0d62752d0ff8fe116b97de",
        "result_sha256": "be58f008e5e65d9858f4c3616d1a54a44c13a9a6ecdfd5eaf7c0de501a82ce9e",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "c17bc04f17594d9e9acf4b4ce7357d51",
        "job_sha256": "19d9f5842f70bd6fde83efbff2872ec38582bc7ae69e2d0528effd16e187e03e",
        "result_sha256": "bdf590a00e8e445f0325c83f24e16ae3e529c60192e5bd2e7002d7d2196ef483",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE16_TERMINALS = (
    "c0468ae4cc19065d4f8f6fc8c0e92e62524bcaec62eb2302a8b5733c2372eded",
    "f37adbe6ee937f470dc7b0ea450be5d8922d3a34688134c6612953681786e1f6",
)
RELAY_6101_WAVE16_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "9120a4377c264d898c2dfdaa68be28bf393bd068053260da31def3cfd212f425",
    "attempt_00.replay_authorization.json": "1059bbc3f4a26dbd8c43bfe6d21e16f384466a783eb3e36357b9ae2a0ac985b8",
    "attempt_01.replay_link.json": "2ef5c7ccf7983860781c80c2cc2846dfc8f84daa1f7c7f31b3ab7871e129e903",
}
RELAY_6101_WAVE17_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "076fd6a9-c824-405c-b5a0-9f7ebec2e1be",
    "response_id": "resp_0b1ed3fb81c788a1016b088f581da487d186a38ed538d35102",
    "conflicting_response_id": "resp_076fd6a9c824405cb5a09f7ebec2e1be",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 211.01859904197045,
    "elapsed_seconds": 211.68580766697414,
    "sse_milestones": {"response.created": {"elapsed_seconds": 31.684189249994233,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 31.68774904194288,
                          "interval_seconds": 0.003559791948646307},
 "response.output_item.added": {"elapsed_seconds": 31.691758417058736,
                                "interval_seconds": 0.004009375115856528},
 "response.output_item.done": {"elapsed_seconds": 47.924547374947,
                               "interval_seconds": 16.23278895788826}},
}

RELAY_6101_WAVE18_CONTRACT = "3ef1cdc126867d2ded8d926965bd80f70d8c3687a61270ee2e402ff7a058e30e"
RELAY_6101_WAVE18_AUTH = "6217b415f93b219bd903da1f75c703c216ec1e61a68452dd45d9f452683b5ea9"
RELAY_6101_WAVE18_PROOF = {
    **RELAY_6101_WAVE17_PROOF,
    "previous_request_id": "5e5ce772bd6ee11f1b2362784562e1d683f48f3dcbca9ad83d049b0ad8fc9afe",
    "previous_terminal_file_sha256": "e644e1eb54cb464a4e4f449fb078e805f55f38d5a19a554565c45990d9440b00",
    "previous_transport_progress_file_sha256": "e35c4092a7cc76dfd64af41353d41fb14f49584ff7412545d80954f3cae7371c",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795725073.1359603,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 163},
}
RELAY_6101_WAVE17_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "c93f82325eef449d965c6cd07d7370d1",
    "job_sha256": "f2d2c713536ee18ee9971e391faa150063d3299d338829f85866b1ed782b0589",
    "result_sha256": "87e159b75f44d519990e2ea50ad519f911c88346214a1489b42e63aef30b56c8",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_6101_WAVE18_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.added",
    "sse_event_count": 23,
    "model": "gpt-6.2-astra",
    "provider_request_id": "28af9e2a-3e94-44b6-8a70-3266dce9c85b",
    "response_id": "resp_0311909841327144016b08919e406887d282946c24629f2c82",
    "last_sse_seconds": 195.7894462090917,
    "last_sse_activity_seconds": 195.7894462090917,
    "elapsed_seconds": 1095.799670791952,
    "sse_milestones": {"response.created": {"elapsed_seconds": 29.485547916963696,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 29.486534500028938,
                          "interval_seconds": 0.0009865830652415752},
 "response.output_item.added": {"elapsed_seconds": 29.4872592499014,
                                "interval_seconds": 0.0007247498724609613},
 "response.output_item.done": {"elapsed_seconds": 47.20936862495728,
                               "interval_seconds": 17.72210937505588}},
}

RELAY_6101_WAVE19_CONTRACT = "caafe807a20d37213b06c1880cc7065834a971c7588fb21b26f6a98f44c54da0"
RELAY_6101_WAVE19_AUTH = "d07b681c836770cc9f5943ad16d6863cfe6d1b3bb2dc587581e96292be3f7208"
RELAY_6101_WAVE19_PROOF = {
    **RELAY_6101_WAVE18_PROOF,
    "previous_request_id": "c912c28c98bf1ab461cb5653076104ccaf929918f0aa91bef73a031b11b41996",
    "previous_terminal_file_sha256": "3b18be441d41efa746520a1c3734a79d36a3b9b0422fe85fd7ebf13f81a239a6",
    "previous_transport_progress_file_sha256": "c30806f0431fc6cf55e2932f7162cb8e7598eb3b173ef9251d1246dffb28e019",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795725649.7751193,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE18_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "a25420736ed14156b62c45aa63147857",
    "job_sha256": "4059f593b6e6917069905a47a867de2b563dd2b7bc1a57cfe45a82755bfd43d5",
    "result_sha256": "60b22555f35a6e6628ee7cdb1a04e744cfcca60806f289a7474a87356711f204",
    "error": "response_id_mismatch",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_6101_WAVE19_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "aa7814f3-9ae2-4821-a0fd-0b5c36f0b8ea",
    "response_id": "resp_0f163772d4233cbf016b0897d1b23487d184a585e7ad294752",
    "conflicting_response_id": "resp_aa7814f39ae24821a0fd0b5c36f0b8ea",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.2267327089794,
    "last_sse_activity_seconds": 195.2267327089794,
    "elapsed_seconds": 198.17091620899737,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.134589292109013,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.13622883404605,
                          "interval_seconds": 0.0016395419370383024},
 "response.output_item.added": {"elapsed_seconds": 18.13797491695732,
                                "interval_seconds": 0.0017460829112678766},
 "response.output_item.done": {"elapsed_seconds": 27.2943535419181,
                               "interval_seconds": 9.15637862496078}},
    "network_phases": {"direct_ipv4_http11": {"duration_seconds": 18.065469166031107,
                        "observed_seconds": 18.07537891692482,
                        "state": "complete"},
 "direct_ipv4_preflight": {"duration_seconds": 0.7129365000873804,
                           "observed_seconds": 0.7251982090529054,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 0.7129365000873804,
                             "observed_seconds": 0.7229494170751423,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.21449062507599592,
                                     "observed_seconds": 0.22649749997071922,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.2253213340882212,
                                 "observed_seconds": 0.4546462919097394,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 13.56521450006403,
                     "observed_seconds": 18.073338417103514,
                     "state": "complete"},
 "send_body": {"duration_seconds": 0.8663502091076225,
               "observed_seconds": 4.506517709000036,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.00022504106163978577,
                  "observed_seconds": 3.6397787500172853,
                  "state": "complete"}},
    "first_phase_seconds": {"http_headers": 18.07646029209718,
 "http_phase": 0.22649749997071922,
 "request_dispatched": 3.493617249885574,
 "sse_event": 18.134589292109013,
 "transport_failure": 198.17091620899737},
}

RELAY_6101_WAVE20_CONTRACT = "f13bc63f8695256d37ae9e742819cd764a415989eb1298b30a4f97b68d77aecb"
RELAY_6101_WAVE20_AUTH = "f132ae5cafe4d1219151c232d1b854cc6dd7fc24534fe1210d723bab28757ea2"
RELAY_6101_WAVE20_PROOF = {
    **RELAY_6101_WAVE19_PROOF,
    "previous_request_id": "c20cdba9520247c56e56bf545fd4d321de60be193086b2040af2e0e3fc8af0b9",
    "previous_terminal_file_sha256": "157563bb66d5aa09c884b71f2a2287585a147915426ba5fc541845a3121d0dab",
    "previous_transport_progress_file_sha256": "be0ca2e33ffd0feeb44d6206b7f9c311de670115b4488293cb105f07e9aa9977",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795727232.2344866,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE19_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "ab403f7d4a1641118e5f70e701ae33f3",
        "job_sha256": "4e168b730f4f87b05e020379d6bcee445d69dc321b2bf1bc9678b1ff45df9355",
        "result_sha256": "f77b0444ff8ec9f44957305b3fa357673a717f9c725c85bbad3ed292ad747ce5",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "57d10f4bffd649b7aa2c50065d1efd78",
        "job_sha256": "b89626bf11297e9ff26711cd3cb0a3e48d45d2c0b6ccdc8c889bf8d9c849a6ca",
        "result_sha256": "1d5bff315f60b87d126009b55f9cf3f1009d7b89a9ce5f77818664bdc0d21c19",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE19_TERMINALS = (
    "8240975e329e0bc59daf21b1bda05287e71f4b61dfaad635225da802ee56f6e1",
    "157563bb66d5aa09c884b71f2a2287585a147915426ba5fc541845a3121d0dab",
)
RELAY_6101_WAVE19_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "043cd54edde70fdef5a116464bd352695d07616e2dca24b2acdd6fb669ea5209",
    "attempt_00.replay_authorization.json": "1e19f8ddd5fd6ff51cb6ae1d0790a841485517e5412f4cd8e29162854342a9aa",
    "attempt_01.replay_link.json": "03b57e1738b5fe7e4f02ed95f05b845abf34b3f336636102027ba681a007bfbe",
}
RELAY_6101_WAVE20_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "69e231d6-ad02-4895-a3ee-7599c8fc4ee2",
    "response_id": "resp_0e21f1eae3a3dc7a016b089df4968887d19ab7ad892c5f5f75",
    "conflicting_response_id": "resp_69e231d6ad024895a3ee7599c8fc4ee2",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 198.82416954101063,
    "last_sse_activity_seconds": 198.82416954101063,
    "elapsed_seconds": 208.37481295806356,
    "sse_milestones": {"response.created": {"elapsed_seconds": 28.345941541017964,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 28.347996375057846,
                          "interval_seconds": 0.0020548340398818254},
 "response.output_item.added": {"elapsed_seconds": 28.349738458171487,
                                "interval_seconds": 0.0017420831136405468},
 "response.output_item.done": {"elapsed_seconds": 45.84804979106411,
                               "interval_seconds": 17.498311332892627}},
    "network_phases": {"direct_ipv4_http11": {"duration_seconds": 28.338568459032103,
                        "observed_seconds": 28.34198641614057,
                        "state": "complete"},
 "direct_ipv4_preflight": {"duration_seconds": 0.9083195000421256,
                           "observed_seconds": 0.9142159160692245,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 0.9083195000421256,
                             "observed_seconds": 0.9119096251670271,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.23898166697472334,
                                     "observed_seconds": 0.24294458306394517,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.24755812506191432,
                                 "observed_seconds": 0.4930660000536591,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 24.252684667007998,
                     "observed_seconds": 28.336896458175033,
                     "state": "complete"},
 "send_body": {"duration_seconds": 0.9519454170949757,
               "observed_seconds": 4.081764833070338,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.0001302498858422041,
                  "observed_seconds": 3.1295068750623614,
                  "state": "complete"}},
    "first_phase_seconds": {"http_headers": 28.3435335829854,
 "http_phase": 0.24294458306394517,
 "request_dispatched": 3.127337541198358,
 "sse_event": 28.345941541017964,
 "transport_failure": 208.37481295806356},
}

RELAY_6101_WAVE21_CONTRACT = "6d4c68b8234db17c79c0b2626e90fe6c29178c42b1aa45da0d88e5ba5a8d7fbf"
RELAY_6101_WAVE21_AUTH = "9f31bb918783ef8f1d1ece1be1c2bd887e73b8322a4176d46fd4fd00513725d4"
RELAY_6101_WAVE21_PROOF = {
    **RELAY_6101_WAVE20_PROOF,
    "previous_request_id": "65fc4b7238cebd3a256d204cc58e21e0843038a229e24f4b568da2811a4a6bf0",
    "previous_terminal_file_sha256": "479980cf2f24c2f29b490a0874ace6e807a41e4c3fbf026e4421c77cdf50c7d8",
    "previous_transport_progress_file_sha256": "fdf40eb4625389acd65f895aff58bcb8c5ec5ef1ef720d7494c32538f58423c1",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795728250.404925,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE20_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "94ff77fc975a486bbc6123e0eb12a505",
        "job_sha256": "3ba4d25064a9b19f41852219238935475d2494c7e71152bc2abe9061053d0439",
        "result_sha256": "d403e2ef21db87d95d17cd4104ba21e612c81ec26858127d0fb1867ba4f50173",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "7d19e415a09c4786858a353cab95ee30",
        "job_sha256": "e152b8f1833386671086a210b3496bce13eb72829d1fefcfb9fc6e3ce924424c",
        "result_sha256": "379e4e50356101777c624c2f7c9d378dffafcd5eb655ed855175b89f3d102ac1",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE20_TERMINALS = (
    "76fa8a43fc6de6b13deb5c86eacf9f5ea464fd7e10bea1fbf3fb651b5d47592d",
    "479980cf2f24c2f29b490a0874ace6e807a41e4c3fbf026e4421c77cdf50c7d8",
)
RELAY_6101_WAVE20_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "91a84f0a64be6398a32ca1256dbc06a876d78833ce70443456c99a590a65eda7",
    "attempt_00.replay_authorization.json": "700129167dc6339227b90ffce83b2db887600f6d5ba9e4f1ca47586d66468cec",
    "attempt_01.replay_link.json": "77be623c93c7e72ccbedcf6c9b540e5097006fd7d737d47ece4859a91eab0387",
}
RELAY_6101_WAVE21_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "83146006-5842-4829-9801-14e67f4afa32",
    "response_id": "resp_0d9c83b16628e94b016b08a1fb64c487d2964040c60e9e1060",
    "conflicting_response_id": "resp_8314600658424829980114e67f4afa32",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.67230829200707,
    "last_sse_activity_seconds": 195.67230829200707,
    "elapsed_seconds": 198.37909508310258,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.436576958047226,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.43958095810376,
                          "interval_seconds": 0.0030040000565350056},
 "response.output_item.added": {"elapsed_seconds": 18.441653833026066,
                                "interval_seconds": 0.0020728749223053455},
 "response.output_item.done": {"elapsed_seconds": 27.749191500013694,
                               "interval_seconds": 9.307537666987628}},
}

RELAY_6101_WAVE22_CONTRACT = "c4d6f97a21f6acde7f1279cba678fd0c9e0bce7e122cfd5fc859c0cda3dff3de"
RELAY_6101_WAVE22_AUTH = "e3b51bc0b106cb3f30c436472da5744454b6dc1688f5d172d4fcacc2eb39b624"
RELAY_6101_WAVE22_PROOF = {
    **RELAY_6101_WAVE21_PROOF,
    "previous_request_id": "e60ce7b55b14a635a97022530ebc5902464ba634192022644732fec8292553aa",
    "previous_terminal_file_sha256": "2eaec0f27a87734206b9ecf09953df388a03e8fecf9de3938d428aa9353f56a1",
    "previous_transport_progress_file_sha256": "c0ca9b967dd2c65ff23c7617b45d9762555cb9307f39b4d6fdeb19e8dd9007e6",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795729309.1526127,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE21_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "ed0e9772e418412fbb46cf4c8adad047",
        "job_sha256": "1461923e08d5c987a455b9c53efa7202cdbd922772e4a2d92a3c3388a4df0d23",
        "result_sha256": "34e09ebe39b0660ab1bb9810282b4f18325aa533716ee7b8a75c49c0d7b5c7fc",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "d61ca71a6ba34c72815eda95a4902d22",
        "job_sha256": "ac1ac4fcfe9ee11f8e47e29c514c99e9ee2bc1c10ef98f20862d1432253e78a9",
        "result_sha256": "61f4095b9cb65ac097354d46eb361c41fbc605b96b8d9b353fefc22e9db7a83b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE21_TERMINALS = (
    "dc2fa0c5acf9e6b4b67d63fec16d54598cbfa88eb8876c8248df336e0623f4b9",
    "2eaec0f27a87734206b9ecf09953df388a03e8fecf9de3938d428aa9353f56a1",
)
RELAY_6101_WAVE21_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "37393cf53dff08c8a023505b64fc1721f30457dd9da7b74f17529097abd23ab9",
    "attempt_00.replay_authorization.json": "c0cbc7a2c1a9623c52b150334a2e7fbaf7b2f3c3866778a65342bce52baf6f0d",
    "attempt_01.replay_link.json": "03b4ae57ba5cbb66135628db23acd709757d11af8ec6f4d2171a18da98af77d0",
}
RELAY_6101_WAVE22_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "3387dd77-b91a-4820-b19f-6f6e0de255d7",
    "response_id": "resp_0f78f455098ebd82016b08a6199b5087d2b3cb3c0f8310bca3",
    "conflicting_response_id": "resp_3387dd77b91a4820b19f6f6e0de255d7",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.45068449992687,
    "last_sse_activity_seconds": 196.45068449992687,
    "elapsed_seconds": 199.22330374992453,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.283509582979605,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.285732958000153,
                          "interval_seconds": 0.0022233750205487013},
 "response.output_item.added": {"elapsed_seconds": 19.288397416938096,
                                "interval_seconds": 0.0026644589379429817},
 "response.output_item.done": {"elapsed_seconds": 28.753125875024125,
                               "interval_seconds": 9.464728458086029}},
}

RELAY_6101_WAVE23_CONTRACT = "43cf05aae93480ca623c10032d3a5820a33cf0f617e10a3bb7115e4570c4b292"
RELAY_6101_WAVE23_AUTH = "062791ed190d3e0c41e9e71d235eb3e225206458597b27ec3c4228f6278121ec"
RELAY_6101_WAVE23_PROOF = {
    **RELAY_6101_WAVE22_PROOF,
    "previous_request_id": "6c8eba5102c55312a8c24941179a119eb843a392d46af3fc5f703a3127e9d152",
    "previous_terminal_file_sha256": "b8a3797b80079c6d75de4fd998461a7ace2c0ccfbb3f034a10379d6c6a2b75a8",
    "previous_transport_progress_file_sha256": "08ec4a8a3b1e3875128876f1c0b28746572aa8fe386f0f3b89c755517092e1d6",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795730260.231831,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE22_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "d2275b39cbea4e1ab69ed8c951bf489d",
        "job_sha256": "9895a2be9a9e4df1f63b96abfb84cc4f3d5ab57f74614697d22ddf14cca7f533",
        "result_sha256": "47fe579e9fa4ccc5e5f51785d641aae08cb69769f6478e4e6d246675b7149d6c",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "d6530498d3dc436e8b05b2dbd49f74cd",
        "job_sha256": "405a8fbdc8ef2f9e8bea2f0b11699f2d6730a2b3e865f4d080d59f438e72ec0b",
        "result_sha256": "034af23b0eb61133d21968855d206edb36202c0dd795da57362b4e9878e10adf",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE22_TERMINALS = (
    "6d294a04c3b4599a7e47e966ff0dfd719963167fb4d9030248578cdbbef5e6a8",
    "b8a3797b80079c6d75de4fd998461a7ace2c0ccfbb3f034a10379d6c6a2b75a8",
)
RELAY_6101_WAVE22_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "ef09e775423c9517b6543d7adfe61cd644248cea10c9f8d0f5cc279c019b0a17",
    "attempt_00.replay_authorization.json": "15171368767598339210258575e06ef7e2480c709997b202408c1389579b7e9f",
    "attempt_01.replay_link.json": "5aa37eefa70e023f6ea16f2de51de80f582b98ddad73236f3a2d839f3c5f0d81",
}
RELAY_6101_WAVE23_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "d99d61c4-faee-4ddf-ab28-0ba80446db4d",
    "response_id": "resp_01f3239edf5c0689016b08a9c4647087d09954ecd17ee242ff",
    "conflicting_response_id": "resp_d99d61c4faee4ddfab280ba80446db4d",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 206.12672008317895,
    "last_sse_activity_seconds": 206.12672008317895,
    "elapsed_seconds": 211.91308645810932,
    "sse_milestones": {"response.created": {"elapsed_seconds": 31.908563708188012,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 31.912377125117928,
                          "interval_seconds": 0.0038134169299155474},
 "response.output_item.added": {"elapsed_seconds": 31.915689000161365,
                                "interval_seconds": 0.003311875043436885},
 "response.output_item.done": {"elapsed_seconds": 48.24363920814358,
                               "interval_seconds": 16.327950207982212}},
}

RELAY_6101_WAVE24_CONTRACT = "ceb747c1291e16f5814e28c17fc2dee22bc1e78119081df755f2209f38070453"
RELAY_6101_WAVE24_AUTH = "149d6c74db271279d33c0e35e669c196fb6317785beff5943c310425922a2bbf"
RELAY_6101_WAVE24_PROOF = {
    **RELAY_6101_WAVE23_PROOF,
    "previous_request_id": "66c41ac9e6e77ba34bb32059c59dd2382a71b5aa955826f5c4b6956ae4f02e7c",
    "previous_terminal_file_sha256": "9090345385a0b06099ecb705e2b030f95a7794879cbad377a197a94560b1a0ff",
    "previous_transport_progress_file_sha256": "104e0ca02ca392e01004728ff656b032b4bbe592bbe9e79161052df61fa87377",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795731182.7983212,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE23_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "f02ae69a17f84c60948f9686d5e9c494",
        "job_sha256": "56f9fd279b72dff4e755c3ec7e8d0e0d91ade05923db5f736bd1230dab188f2e",
        "result_sha256": "36e1a958d5a40f45b107dba36c5894721321a0f07653024650172120f71d8ad0",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "390ae9ee72354d5dbeaa69559af7765c",
        "job_sha256": "892572bc928a0202e5b8b0d535da26fefe83114a96e46193533f69b9c969964f",
        "result_sha256": "a300bec6b36c4de1a0790caa78e03e2745a13a41ef870596f7c89380236b3370",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE23_TERMINALS = (
    "5e37437642c86d92ae986740a9d9b960526fdfb66357c308854bb6607f11004e",
    "9090345385a0b06099ecb705e2b030f95a7794879cbad377a197a94560b1a0ff",
)
RELAY_6101_WAVE23_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "f72bb313c783b9af255d6ec92b383f5df43d6d9a7d7f6ec665ad830167ddff64",
    "attempt_00.replay_authorization.json": "1a451a95cecc32c215d00058b7c8a5e5a6f25d20dddac165d5fe815be6cf42da",
    "attempt_01.replay_link.json": "5a8eebbb1c534842c9c3ce0cfce197d77d1a80bba2362f90f518c234fb4d3472",
}
RELAY_6101_WAVE24_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "32bff7da-8bec-48ed-95bc-1ddcfc8e6734",
    "response_id": "resp_00b39143f7411dad016b08ad7069a487d284446055f038c8ff",
    "conflicting_response_id": "resp_32bff7da8bec48ed95bc1ddcfc8e6734",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.0334075000137,
    "last_sse_activity_seconds": 195.0334075000137,
    "elapsed_seconds": 197.85632333299145,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.906683332985267,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.908817833056673,
                          "interval_seconds": 0.0021345000714063644},
 "response.output_item.added": {"elapsed_seconds": 17.910795415984467,
                                "interval_seconds": 0.0019775829277932644},
 "response.output_item.done": {"elapsed_seconds": 27.154793333029374,
                               "interval_seconds": 9.243997917044908}},
}

RELAY_6101_WAVE25_CONTRACT = "84af4f3bfdacf0ac392ccfb4baef885aff540a6f61f8d32ecefd95cb7e8812f7"
RELAY_6101_WAVE25_AUTH = "c864fff42673d80550b8892036dcfbb3fa4bfecd6f06703cf396e61f2f9dfd8f"
RELAY_6101_WAVE25_PROOF = {
    **RELAY_6101_WAVE24_PROOF,
    "previous_request_id": "bf798a79f9fad7dd0f7fde713d36b244b5fd1ca0fff78ee94b666f0fffd6d7f9",
    "previous_terminal_file_sha256": "4fc6c4ec55b423452946cebb494f6fe714c66f233811d12dad50f4acfd4b57a5",
    "previous_transport_progress_file_sha256": "e21016b2345f40a5da7297ce0ddce70896568b977aac85ad7fa6e483c8dc3fa4",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795732175.9664137,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE24_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "07394a0e514f4d25b4af9676265111d0",
        "job_sha256": "53941a1e60ae4b4bba55ba54ac0a16f398aca7fe955f2fe92a8bc0d5249d8f2a",
        "result_sha256": "ab2c4a5ffe46ba68af87d34e3c53ec30d36e44ff3236fd7e6f1e92b2a5fc106a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "b4d19df55bc64adda0c7e946a472ded2",
        "job_sha256": "ea0510d24ae2ebac218d8e768cb4e940a9eaeb91d32c027c81e14fcb60cd6464",
        "result_sha256": "8882b9e73a658c3bc172f0eb3f1b2086a49f7fc0e121e06f91cd06f74dc87ebb",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE24_TERMINALS = (
    "7127ac8c06772478fc849f15e47b7a24723a9417acfe4c02b7177423d5e15b83",
    "4fc6c4ec55b423452946cebb494f6fe714c66f233811d12dad50f4acfd4b57a5",
)
RELAY_6101_WAVE24_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "70b5a76e718e533a3547e7b7a4c44ad0837116a451f1f4607f13e13f8b8925aa",
    "attempt_00.replay_authorization.json": "12f0873b988baa68b679f578f79189eff14e23568db1920f405631f444e39c8d",
    "attempt_01.replay_link.json": "03c6b44c7a522e1d35f13aea6c41e4c3531b36baf35f5030a39e8d21b3f2f444",
}
RELAY_6101_WAVE25_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "e365fc97-2298-4e00-a060-a98f7500e4ed",
    "response_id": "resp_01aea0054f8fa5bc016b08b145f90c87d08ebb6bb90b70f56f",
    "conflicting_response_id": "resp_e365fc9722984e00a060a98f7500e4ed",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.44041116698645,
    "last_sse_activity_seconds": 195.44041116698645,
    "elapsed_seconds": 198.06330612511374,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.080255625071004,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.082239625044167,
                          "interval_seconds": 0.0019839999731630087},
 "response.output_item.added": {"elapsed_seconds": 18.085203167051077,
                                "interval_seconds": 0.0029635420069098473},
 "response.output_item.done": {"elapsed_seconds": 27.39533062512055,
                               "interval_seconds": 9.310127458069474}},
}

RELAY_6101_WAVE26_CONTRACT = "4deacbd298b90031526a2f72fa696ad36c122892ef98d9486fdebe85dc1ea9c1"
RELAY_6101_WAVE26_AUTH = "77456d673746ebb844959eb8df407dcf54acf41a18613640603b2263cb23bacc"
RELAY_6101_WAVE26_PROOF = {
    **RELAY_6101_WAVE25_PROOF,
    "previous_request_id": "0e964858f0c082d764f49428d11c27adbd173c0614863b5fb3cc20f656e9868f",
    "previous_terminal_file_sha256": "29c511950282e9c37e8b81240e0415e7938d38afb47ea81720ef8819ec6216ee",
    "previous_transport_progress_file_sha256": "df1ba478f9b1465f8758d5a3a8a643c3ed6f08b3e0bb45f5e45c9b1255bdcd96",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795733263.9176679,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE25_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "4316b74f72db4b33a21bc1829c71fc89",
        "job_sha256": "7aa1d7149e1a677c3a80fbdfbd7327728f38bc00d7ab093bc4c74dfad33f875b",
        "result_sha256": "ea388e78da8b866825606ce82171eda04b41669f8055efcc8ee2f77b2e8433f7",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "cccb5a1806a948929b171a8d429bda3e",
        "job_sha256": "df17b532b6f2c0dd6d387da730e8300440ce76df167b76eb5de47db56a21d74e",
        "result_sha256": "c766475970abbfd17d9fdfafb7508e21eda5ba605306e4c841d3f15429503e45",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE25_TERMINALS = (
    "d05f4ecf35feb3e742ed8e307a64a51afdf0d838a66c69071bbc0fd64df0c299",
    "29c511950282e9c37e8b81240e0415e7938d38afb47ea81720ef8819ec6216ee",
)
RELAY_6101_WAVE25_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "af03fdac3a57b5cb839fb4f27fd5e8b3332183244d48a50b5de6ef75b8116db6",
    "attempt_00.replay_authorization.json": "0d05d0fb16c6d03c9f8a6b17cc5b8eaaeb44475b0896aa4a8f7923aefd50487e",
    "attempt_01.replay_link.json": "c75335c51cd191733543f66702fdf7dfd346f2c875d25381f8888c8e76e70bd8",
}
RELAY_6101_WAVE26_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "02a566e1-06ef-446d-8990-6d76f8ca1c5a",
    "response_id": "resp_09b624202fd0204c016b08b58f07f887d0a3a8538c84610b34",
    "conflicting_response_id": "resp_02a566e106ef446d89906d76f8ca1c5a",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 198.5791477910243,
    "last_sse_activity_seconds": 198.5791477910243,
    "elapsed_seconds": 201.24177045794204,
    "sse_milestones": {"response.created": {"elapsed_seconds": 21.288914707954973,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 21.29085245798342,
                          "interval_seconds": 0.0019377500284463167},
 "response.output_item.added": {"elapsed_seconds": 21.293740124907345,
                                "interval_seconds": 0.0028876669239252806},
 "response.output_item.done": {"elapsed_seconds": 30.601800665957853,
                               "interval_seconds": 9.308060541050509}},
}

RELAY_6101_WAVE27_CONTRACT = "87c095a9fe5c0c742da891ea3da33a732dc1bd06bfaaed144c7bb81604074ebb"
RELAY_6101_WAVE27_AUTH = "3804f67b2e76d53332b3357c8db888f26f9e2298673716aa597308001072b95e"
RELAY_6101_WAVE27_PROOF = {
    **RELAY_6101_WAVE26_PROOF,
    "previous_request_id": "8b144714e29f0fd8fca18b02bf3ed4f320aef4d740dd81eb27f5773021bcb1a5",
    "previous_terminal_file_sha256": "6ba2c8bc2cfb91026ddacf70fc240e0db538d5d136dfc0e30f219732d2c554a2",
    "previous_transport_progress_file_sha256": "346d5eb3a4bfb93ddf5a975ffac3a23b0d78876b6d19d6f9d9f2fb74a650fe0d",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795734534.2416294,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE26_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "2f0b0d68e2c746b29d99f3d995092673",
        "job_sha256": "561304819eb5838261fd86b786f528104c10d786fc1c00c1537b6987f840e4ab",
        "result_sha256": "067c490172a55e7f65ec3f8fea2b237393579f7d2173641911b59e356206b324",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "792e6c2a9c41483ab2ecdea69057b635",
        "job_sha256": "aecfcc46008d09d4c33d271af1d5597e341a5b75f5e500c9c08e9e3d8179b687",
        "result_sha256": "d4d1efd3188bb20b3944e6cc5d465861146329375685d5d167b53f48d876e4df",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE26_TERMINALS = (
    "9889f4c2c65df4ffd3c0ec07a94ca5c3bf5343a3a299a50f972068f3e60236ad",
    "6ba2c8bc2cfb91026ddacf70fc240e0db538d5d136dfc0e30f219732d2c554a2",
)
RELAY_6101_WAVE26_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "15f6efac4b7d9760e313234b55162b987a65cbd3d6bb1cacb027d1a80ad2ccee",
    "attempt_00.replay_authorization.json": "b4721185709c55a35b7fe8d79ea0297947691cf100f5c2066bf3042dc1f09fa8",
    "attempt_01.replay_link.json": "4db3a9c432113be8f77f4cd5afc8f5d28d756b5307f3b3f4fa6b33b5b45b444e",
}
RELAY_6101_WAVE27_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b49bf319-cfce-484d-a5a9-0c8634345977",
    "response_id": "resp_049994cf5ff52009016b08ba83e0e887d2899d4da4af066dcb",
    "conflicting_response_id": "resp_b49bf319cfce484da5a90c8634345977",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.8672819591593,
    "last_sse_activity_seconds": 195.8672819591593,
    "elapsed_seconds": 198.6679510420654,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.739110042108223,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.740597125142813,
                          "interval_seconds": 0.0014870830345898867},
 "response.output_item.added": {"elapsed_seconds": 18.742711042053998,
                                "interval_seconds": 0.0021139169111847878},
 "response.output_item.done": {"elapsed_seconds": 28.471411042148247,
                               "interval_seconds": 9.72870000009425}},
}

RELAY_6101_WAVE28_CONTRACT = "d482a87af7a5b4340739d810e50c76b8f3d0abae590aca107b5a5a8716248a67"
RELAY_6101_WAVE28_AUTH = "c8513c9d0350e8bf0a32b83121fe81421857b05cc8e4aa3f4d42fd959387f86b"
RELAY_6101_WAVE28_PROOF = {
    **RELAY_6101_WAVE27_PROOF,
    "previous_request_id": "880a99d18babd190e20db6e6b2084a282c92d7277ef742fc919abb2155f73178",
    "previous_terminal_file_sha256": "ae92f952b59484fd9065446c7eaa2d95e9d5a1e86c6365b5ecde59d320e96441",
    "previous_transport_progress_file_sha256": "c8ed8413fb8c5c4ebbd97ac8404444994e69b4e5b3a7683d694e7e4adbede73c",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795735503.443264,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164},
}
RELAY_6101_WAVE27_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "c5f6778ed0f4440086cb94ee949de396",
        "job_sha256": "3f15042b212178c91a793e150134e628a18f89e2b6d9575f96e8c2b95cee7d09",
        "result_sha256": "ddd1ffe8c3680a49fd2a22e2756b1f0b1cd0317b59d7146e41c8301b70777739",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "bbb02e67ce07472fbeda336256431039",
        "job_sha256": "7030877099c11680852be2c3c269c4267ec45424bc01528bf3987ac5f8a8e690",
        "result_sha256": "694e82fbd66813e7b6b4d248907ab68abbca529306ded1bb029961335f38b7aa",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_6101_WAVE27_TERMINALS = (
    "c08b0ebe90c475d96ea48d4753f99b803cf7ceed310d51b7adecd362785ae16e",
    "ae92f952b59484fd9065446c7eaa2d95e9d5a1e86c6365b5ecde59d320e96441",
)
RELAY_6101_WAVE27_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "ea47aa2857cb30874ba7b4897d6e229efd18b9c7f6964b65013c68e875af5427",
    "attempt_00.replay_authorization.json": "be971b4b76e428463003b455314fc420e3625b8b306bc5fd0497ec2f26078ea7",
    "attempt_01.replay_link.json": "2d974089c706923f1a979778cd2dee64364b49cdf45c1f73b63ca8d5c49e6e81",
}
RELAY_6101_WAVE28_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "044df5cf-937d-4548-80e8-257be0a61f2b",
    "response_id": "resp_03ddae8968243ba5016b08be4d3db087d1beb5bb84492ad391",
    "conflicting_response_id": "resp_044df5cf937d454880e8257be0a61f2b",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 198.20465458300896,
    "last_sse_activity_seconds": 198.20465458300896,
    "elapsed_seconds": 200.8405782498885,
    "sse_milestones": {"response.created": {"elapsed_seconds": 21.210311708040535,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 21.212981541873887,
                          "interval_seconds": 0.0026698338333517313},
 "response.output_item.added": {"elapsed_seconds": 21.215132207842544,
                                "interval_seconds": 0.00215066596865654},
 "response.output_item.done": {"elapsed_seconds": 30.203939375001937,
                               "interval_seconds": 8.988807167159393}},
}
RELAY_6101_WAVE28_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE27_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE26_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE25_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE24_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE23_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_6101_WAVE22_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE21_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE20_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_6101_WAVE17_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_6101_WAVE16_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_6101_WAVE15_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE14_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_6101_WAVE11_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_6101_WAVE11_SOURCES = {
    **RELAY_6101_WAVE10_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "ca1903c5da89f1f234dbef67900df4c21a1f14c59b387d2198b29c3c65259677",
}

RELAY_6101_WAVE12_SOURCES = {
    **RELAY_6101_WAVE11_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "a0be59f66e42d0a048958b1165a0c2c728a576d959aa23d78c01f19543b224b6",
}

RELAY_6101_WAVE13_SOURCES = {
    **RELAY_6101_WAVE12_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "3892823a3a32eb004115eb3e45f27c9a1a7d34b27555073a77327a8099f96666",
}

RELAY_6101_WAVE14_SOURCES = {
    **RELAY_6101_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "26559cd842266dd6893136ad24f23a62643c797ff6d311b78dad688a6874777a",
}

RELAY_6101_WAVE15_SOURCES = {
    **RELAY_6101_WAVE14_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "fb43591fbf273cfbeadf6969acd1555abf065dec981cc703611cf30c9ac21ca5",
}

RELAY_6101_WAVE16_SOURCES = {
    **RELAY_6101_WAVE15_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "d68d3d533a08fd3c6138146ca7d9607198cc09bb88aa1a6debdb973a86d67655",
}

RELAY_6101_WAVE17_SOURCES = {
    **RELAY_6101_WAVE16_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "06d0cc7ac99d13dd139f717afc45776b1296d1f6b0f777e5685b51d51d6d69b5",
}

RELAY_6101_WAVE18_SOURCES = {
    **RELAY_6101_WAVE17_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "584c4f4d3db84a8f2b54bedcd036b3add440c354cfa18a000d6be767c2840293",
}

RELAY_6101_WAVE19_SOURCES = {
    **RELAY_6101_WAVE18_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py":
        "5ea4b2415fc2779471990e5192d9ef9a3be1f5233b20b42f651c034f6d493774",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "e2f5c459a664c24e27c358f53c5aa62dba9b1a336766ce7e8dd79e209bc5c48c",
}

RELAY_6101_WAVE20_SOURCES = {
    **RELAY_6101_WAVE19_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "a6bd7914e0330c13646e603a3c00491ab658b425341acf9b3ec8615bc9b04f55",
}

RELAY_6101_WAVE21_SOURCES = {
    **RELAY_6101_WAVE20_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "5d9f7a81dfc870297718b55926b22a2f80e7bef782cc5316803e7aa339b618c4",
}

RELAY_6101_WAVE22_SOURCES = {
    **RELAY_6101_WAVE21_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "484c78d7fc5882b2405b9fed22e8619abfacf99c5bfb73a906f238aaf27ff8de",
}

RELAY_6101_WAVE23_SOURCES = {
    **RELAY_6101_WAVE22_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "89352f3fbae08734dd36b75c4048133935469724bee14c7f8f031d9d44756dd5",
}

RELAY_6101_WAVE24_SOURCES = {
    **RELAY_6101_WAVE23_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "8c18a3abc9402d724df33f2c89878a56a64e8bc7eb3a16533db84430e95e3fd8",
}

RELAY_6101_WAVE25_SOURCES = {
    **RELAY_6101_WAVE24_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "a9f80ec772df54f5ae25867f2614dc96dbbb364f1b3b33fdf589157474f56c9c",
}

RELAY_6101_WAVE26_SOURCES = {
    **RELAY_6101_WAVE25_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "0d4efad56ffdcc74a00187826c6f75b2297dd028bee28aaf4d1ef022aefe886e",
}

RELAY_6101_WAVE27_SOURCES = {
    **RELAY_6101_WAVE26_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7d43d299d496676dcdfb6f77dcea5fa14cd8e35a4a4e211aa47b7f84a02f9631",
}

RELAY_6101_WAVE28_SOURCES = {
    **RELAY_6101_WAVE27_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "d3f976f7f33fcd5c5d24cf20e93d4fd38d705fc04ab00d4a221ad99a0e66955f",
}
CACHED_R3_157_OLD_CONTRACT = "a135a2a3d10e80f12864be66eb5bb65046ced589f8767969b3f9cddf3f6c006e"
CACHED_R3_157_OLD_AUTH = "dc505def300d7dd7024819d6cb4494e0d042bc6c1fd8ea5971aea90b602e58a5"
CACHED_R3_157_CONTRACT = "f142af55eb54dc184e3cd7d19f16717712605df25044a16c19918a62884ab776"
CACHED_R3_157_AUTH = "17e8e477bba8760b46d6977d279573e234b807a9fdd7c4a637be5dda12dea37a"
CACHED_R3_157_CHECKPOINT = "c8c681846156b44af1ef2b2bbf94cf5492764cd4d9bf321aec106ef507b06cd3"
CACHED_R3_157_PROVIDER_RESPONSE = "resp_0403535e9e00decd016b08039e0b4887d2994dfd88faabbaea"
CACHED_R3_157_MIRRORS_SHA = "b505c2ba7de680aa9f77ca5945c5dc720a9e29c8906ef7118ef1fedf889403fc"
CACHED_R3_157_REVIEW = {
    "attempt_id": "d788b24870c14eb3ae7968c7172244ee",
    "job_sha256": "b901812e38b61ef0f9ca9bfe91d53cc7d35ffd1ea16e39a849451cb31df60da8",
    "result_sha256": "9fa7fdefe0d76a3b8e80709032ca1a6d06bfa4ed925f3d742c60058190c1eca2",
    "raw_response_sha256": "e3abc12a7e91aaa92c14670298b51940c5461f5a303504c8b810b526597fde21",
    "final_text_sha256": "2e2c5140e8c95dc871bccd5f71a39aeb4a95c32e81db0843dd691a007a413233",
    "normalized_plan_sha256": "eaecf42e4679501be50fb4f938224816972317961e58781d05964f0f77c8fc4d",
    "ordered_candidate_ids_sha256": "6b581d414db7d5559a28af69f775b5e0d13f79812a609642c27c76f1644f4acd",
    "archived_terminal_file_sha256": "39edae1cb53b26776fe529166753348de3a46b374a2f52db2915ab0aef6f8d95",
    "archived_decision_file_sha256": "a49dafb6821b789ed51373e9df957ffded1345f769984e3f178881945f9c7976",
    "role_field_projection": {"path": "/choices/18/references/0/reference_kind", "from": "single_factor_contrast",
                              "to": "observed", "observed_id": "O-0959", "candidate_id": "C-2926", "axis": "PE/EA"},
    "provider_post": False,
}
CACHED_R3_157_BUDGET = {"first_logical_requests": 32, "maximum_logical_requests": 96, "maximum_physical_attempts": 192}
CACHED_R3_157_IMPORT_FILE = "ed23e6b33ff06a331031599d8b885d8d233a396d092bc6d68c401d99152d44e0"
CACHED_R3_157_FITTED_FILES = {
    "round_03/turn_00/attempt_00.terminal.json": "b9e5023a6795b1e1a8f6a39a0211250454875fc3d957fc5200169b0b37f82be5",
    "round_03/batch_freeze.json": "c08566979fb97938c67a50a9e66840a7bdf827e65608679ca5578ebeca60dbd4",
    "round_03/selection.json": "ad05e526557e8ab84d9ee0854b6b4e88301c47ef187a2c1ac76a514cec1a3071",
    "round_03/memory.json": "b540d8fc9d10020b5e34b2dc8eec8a68fb0d8b46f12bdb76f1f1394e9e733bb7",
    "round_04/source.json": "bfb9e71acf549811b50c221546a987851c06f42550df5f64838461f0defd0324",
    "round_04/transition.json": "9b6d125b72b9eab66943687213d9d86b934c4c50884e006931a04043b2aff37a",
}
CACHED_R3_157_FIT_AUDIT = "a8b4f5d3cba3817b72f510f75044f77c9d5dcffd3440a799fa42f444c0b5710c"
CACHED_R3_157_METRICS = "f7254caaf87928cbb2d34039952bf275b5eaa8ed92cb2de8219309beda2b59c6"
RELAY_157_WAVE07_CONTRACT = "652e1945d0298f1fc99946cd58e398d15de80020cc72e32af56ce2ff3ffc9016"
RELAY_157_WAVE07_AUTH = "49c86ded7d5c1f26c6d728c8e5877e10cf6866b2abf3d0cf3cda0b25adaf70ad"
RELAY_157_WAVE07_CACHED_REVIEW = {
    "proof_sha256": "84a173798d13de2596449dd53d21052972be058f11b41ac6bdc5b2edbf2fcb36",
    "receipt_file_sha256": CACHED_R3_157_IMPORT_FILE,
    "resource_contract_sha256": CACHED_R3_157_OLD_CONTRACT,
    "resource_authorization_sha256": CACHED_R3_157_OLD_AUTH,
    "provider_post": False,
}
RELAY_157_WAVE07_RESOURCE_SHA = "f06b8cc813aa4124c4be9fe4375ff1eba7594b4e3a7f612e26417c33c535d16c"
RELAY_157_WAVE07_PROOF = {
    "logical_index": 0, "payload_sha256": "57e28171cede075156b2a3331b8322cb994805651134644ab0cadb56e59639c7",
    "previous_request_id": "46ebd031c878dadd645b2161415a45cdf838dafc086c7f45834eb25ac03589de",
    "previous_terminal_file_sha256": "9dbd308f49082eac9bf6e240402eb9886496590fb3c809d27545c72a0bcdb9a3",
    "previous_transport_progress_file_sha256": "4bea8bef10d47c4583368202fe8ca34e1c0dffa7d34902d7ba368667a9323f92",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795692500.964523 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187},
}
RELAY_157_WAVE07_PRIOR_ATTEMPT = {
    "attempt": 0, "attempt_id": "ed2d0f8f95ef40c9ae73c25f397d5468",
    "job_sha256": "26ed0ad0b1c1f12e9818cb0289dec1ec4c7810903a2e8fad76585c9c7fcbe193",
    "result_sha256": "14552112cb46b531dd6f5990213ad7a9620b1ea14bc3e0c6346737dcd40d9cf0",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_157_WAVE07_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_text.delta", "sse_event_count": 1264, "last_sse_seconds": 1748.327394583961,
    "provider_request_id": None, "model": "gpt-6.2-astra",
    "response_id": "resp_035fa91e2ecafa2a016b080d900bac87d1ba82be06c4badff7",
}
RELAY_157_WAVE08_CONTRACT = "c522b2d38f3f07dfa2eba566dd069bf44c21885518760e5179fa6315615f4435"
RELAY_157_WAVE08_AUTH = "f116399a5c511f2fdf9c064f1d052c8662d635cada4c2ff6a925d9f9703a64cb"
RELAY_157_WAVE08_PROOF = {
    **RELAY_157_WAVE07_PROOF,
    "previous_request_id": "eefcf01bbf10877374e72f79d848310e02a9f1c14e8d17aa9c6921543e567865",
    "previous_terminal_file_sha256": "dde8e232534c842fb25cb2a0bcc8b5ff3951f1703fa0e96696c19e24e7c92a58",
    "previous_transport_progress_file_sha256": "462eedc9bab98efeba02074297c9a51eb32b89cb2a0b6f2373cb78dee93ca47b",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795696600.5165217 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188},
}
RELAY_157_WAVE07_ATTEMPT = {
    "attempt": 0, "attempt_id": "480fca7d77f74e2bbbb8e6bb0bd2429e",
    "job_sha256": "669348c7843eb21d30691c54ab83d4f28b5b70148efddbd717fe9adb6f388035",
    "result_sha256": "07678784ab39ef104e604566806acd5d2e187e88015637b28c7d215391c07625",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_157_WAVE08_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta", "sse_event_count": 2783, "last_sse_seconds": 3561.158813457936,
    "provider_request_id": None, "model": "gpt-6.2-astra",
    "response_id": "resp_0f418a2e94c1bd0b016b0819d504a887d1af55f85e4323df53",
}
RELAY_157_WAVE09_CONTRACT = "d0a49373985c1c6bd255894f7f35f4ca79c6fd59a9a6c939857f87cb788d7b79"
RELAY_157_WAVE09_AUTH = "28eb169d3d948e19d52c6e2dbf951db74c525a908ce2207dc4629783cb93f287"
RELAY_157_WAVE09_CHECKPOINT = "8ddf737cf20560684e4be5bc62f9886006d49369840558c3bd368b6020f96c12"
RELAY_157_WAVE09_FITTED_FILES = {
    "boundary.json": "61cd70dc48adbd2958bd81844bb2e5dec91030d3ea3d55bcf6713dec7ed59e0d",
    "round_04/boundary_user_message.json": "819e36858efbf1dece8620035ccd13728a9f17aa6c737e71d21dda20a7366c34",
    "round_04/turn_00/attempt_00.terminal.json": "64073d8a4cfa0d269b9c1540ac139f678dc9fe9dd821c5c138720097a80b1704",
    "round_04/batch_freeze.json": "750f620462b175b0902e090621bdd69df7926ae67a8e0ddc8fb8906c9c2ed19f",
    "round_04/selection.json": "60e78ac73c0544366f7d777bdb66e19561f77a7eae0703f8d1e6de00c19815ca",
    "round_04/memory.json": "0f1f4667d916a7086a4bb413a0ad9c485cb221ae61d23251291bb81a6373d5f8",
    "round_05/transition.json": "624e3130925174062977c5552e68ec1df1ddbb184309eef96454b5495945e6f1",
    "round_05/turn_00/attempt_00.terminal.json": "f05f175e9f37b29a7c3cb5076cd75f1b356bd70e78c23bf96961d5cef49d6ceb",
    "round_05/batch_freeze.json": "42bfba810e4b1257cc25da446edda28d667ae3ca426c2df24e03d66ab9bee9f4",
    "round_05/selection.json": "ae6d9e24d794d614e9a7e517ac36be4a2369950819db11b28ef6bae89f9c1605",
    "round_05/memory.json": "c3d98a983778ecfabea09529fb2394e3e72758f4e731c7c484a545025d168969",
    "round_06/transition.json": "556978252cb8fa1533ee985a85fcd06ea1ec4ce6aa866ec3fab1df0939acc34c",
    "round_06/turn_00/attempt_00.terminal.json": "78aaed50b975b4bb54db16455d1b2449ca8e4bff31665faa7fc3fcaba96ce784",
    "round_06/batch_freeze.json": "aecc0b4addc25dd58f61b955f882fcfbcbfa6e0591b44c56d67823c5fee6f8de",
    "round_06/selection.json": "8633cb14dce70f801c48066a1dc8004232fe0d3f91313c91b2b66b04ab23ecf0",
    "round_06/memory.json": "2ddffdf429aabee1493cec68248e696aaf52578433bf61308ab2841317cdc897",
    "round_07/transition.json": "26cf746fe168b0942a8aeb55a407b366dba8356b15f50da4041924120307b047",
    "round_07/source.json": "ac42a77b7b18963db4be8ca807aa68bc04636643cdb8b0f9d0226a3f515a1aee",
    "round_07/fit_000/fit_audit.json": "7566158a2562efe3570d2cf43458481a606649b8d2215f3262357e0d482cc542",
    "round_07/fit_000/validation_metrics.json": "2a990a91faedc98ee4e7d214acbed5a6d2de75f6597e9d09565355271bc05b70",
    "round_07/authoritative_packet.json": "767f945a9801c53434e2dea7eb7756d5499cc6b0af3d5f77c2a633c800957793",
}
RELAY_157_WAVE09_PROOF = {
    "logical_index": 0,
    "payload_sha256": "a107c416d5d99771b3b14a88ee77b81c73378807da79cc849c3da05edae9e1e3",
    "previous_request_id": "43d657284193e17a47274bf0660ccdb31dfdb12baa4e91cb686e65ea5ceebfa0",
    "previous_terminal_file_sha256": "21bcafe25908c82563fbc9a2c3fd597142c3671ffdaa9b2b1bf9585735117ad6",
    "previous_transport_progress_file_sha256": "aeab3db281169048857cc9bdb376f946b02e51bcc8a5fd8495a577ea79257f72",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795705081.8307962,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE08_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "c72548bb89114b64b95972fc24b5910b",
    "job_sha256": "fe18ea111eadef8a26bb9f9526fea10dd6459b3c3b4f65f18bff2975fa2df16f",
    "result_sha256": "4db176100825a81b20a949d476a801a957fc4286385add839054fbaf32ac85c6",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_157_WAVE09_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta",
    "sse_event_count": 5937,
    "last_sse_seconds": 2942.470113791991,
    "provider_request_id": None,
    "model": "gpt-6.2-astra",
    "response_id": "resp_0670888a12db153b016b083a3afb7487d18710c78143d1258d",
}
RELAY_157_WAVE10_CONTRACT = "da528d3431e8029204cf1e4bc9f0cb80e26b43824ca8fd2b50b91c61e21e3193"
RELAY_157_WAVE10_AUTH = "779450ec6b1fa6ac85653d79887d8df64a0f7a9e25fc212ac0b2e84c6af43e8f"
RELAY_157_WAVE10_SOURCES = {
    **RELAY_12203_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "10b11c8d68b2b83895b24f7d93d5d20d9b42624d604dc7841dbe85a547f629fd",
}
RELAY_157_WAVE10_PROOF = {
    **RELAY_157_WAVE09_PROOF,
    "previous_request_id": "e59ccde4457fa92dfe70834781d7f769f6c94532db40c62b6dbd658040db2c42",
    "previous_terminal_file_sha256": "75193d8180ad37624473ba3579c45b376aff7df4d830abb0c372476623199c79",
    "previous_transport_progress_file_sha256": "36071445278b32df8d46f8b772cae7c9cde591dd1e5b0f7a135d49819e7cd239",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795707828.740078,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE09_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "ee1c6e1526ee4fd5b8985667fd5a7acd",
        "job_sha256": "1cb274a35e10ad4baa1b410d0ea5fe9e2b25add6c3784f4982b61a65add44612",
        "result_sha256": "e9fcb62f40fc685b20fcd2635a1a1404afe126918f341effb4730f5b9608ef7a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "a36eb5dadf3848c480a6111e30fae29a",
        "job_sha256": "8028423b74bdc556a37ca0a48d59218faad18a5c8db697646dd89f9f2a18e822",
        "result_sha256": "9f59cc0819cefc4ecadb30992a3ff000d83010e75c998336e1aba6eb5cb3f960",
        "error": "timeout",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE09_TERMINALS = (
    "81c73c24dd749e6c148cce60f23cf1466d609619a4b289128c4c532a256e7c25",
    "75193d8180ad37624473ba3579c45b376aff7df4d830abb0c372476623199c79",
)
RELAY_157_WAVE09_REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "5863950fdd116c32d8127d13f10de98debfb08094c808781f4cc200e1486a9a9",
    "attempt_00.replay.terminal.json": "d87205578d47784081525c1119ff44137b4c6e9e7290822dadcf3863c730eada",
    "attempt_01.replay_link.json": "58dfea77b6e9be2212170b4f3b9b7865c481480e594dc7623e46decfae43591a",
}
RELAY_157_WAVE10_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.done",
    "sse_event_count": 16,
    "last_sse_seconds": 355.43698091688566,
    "provider_request_id": "9c1bb9f3-9ba1-4dd9-b8c2-f0ab809c1a13",
    "model": "gpt-6.2-astra",
    "response_id": "resp_064c196a68bb4cad016b084e30589087d09f71396652606830",
}
RELAY_157_WAVE11_CONTRACT = "b3160c0d7152da8cd96c435797608ed5f4ba1599a8bbb25d9abf78e3688f7f72"
RELAY_157_WAVE11_AUTH = "275c02cc9719bd58133cc4518c3872f27eb90695a6ce3d9a6949859379ae483b"
RELAY_157_WAVE11_SOURCES = {
    **RELAY_12203_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "09d13ce1ed73f50398fc823358b5f0ee71d5e1ad857c30ccbe5eb9089d56eaee",
}
RELAY_157_WAVE11_PROOF = {
    **RELAY_157_WAVE10_PROOF,
    "previous_request_id": "7a9a81fe57c993a4d8725a7f19a6e19d71688b80227af45684742361f547452a",
    "previous_terminal_file_sha256": "ef3d7e834b035d862625770e776f7b31d7fa7abfb7cffe2b5cd1b67b34c0de40",
    "previous_transport_progress_file_sha256": "c69ee67ffbec7fe9865edce0b086b6fb6abc810f25c5b128b55e36c960bcdd35",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795709712.7965784,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169},
}
RELAY_157_WAVE10_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "1403a2ad4fd640f38431fc5b41e8c63b",
    "job_sha256": "135ac7ab824e0583c94f2b7eddee0c45e0133a96152fd772d444dd7e962ebf83",
    "result_sha256": "75f28f0d88372241d4cfffdbb8c14c79c35192ecb44eb1c900c4881dfd04ecc1",
    "error": "connection_interrupted",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_157_WAVE11_OBSERVATION = {
    "http_status": None,
    "phase": "transport_failure",
    "error_type": "ReadError",
    "event_type": "ipv4_preflight_http_401",
    "provider_request_id": None,
    "model": None,
    "response_id": None,
}
RELAY_157_WAVE12_CONTRACT = "fbf1de0b1b25fef8003e01ee3b098554f356d150ab005f7ff065e51efb36c7c2"
RELAY_157_WAVE12_AUTH = "706ed123af0f865e22ef4140fbf55df1ba79d2569e42900d6c917fd56b0f1fb6"
RELAY_157_WAVE12_SOURCES = {
    **RELAY_157_WAVE11_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "079d47b0a0b90f94cc1d985169b4d2c2fbd177781527e1fb12abac7e147918a7",
}

RELAY_157_WAVE13_SOURCES = {
    **RELAY_157_WAVE12_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "00b1c8ff747cc32bfd7004a630363e3afe59f6bbec69bfda02a21438c2ca2835",
}

RELAY_157_WAVE14_SOURCES = {
    **RELAY_157_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "f1a2d60801099a22eb20c116ee539defbb719e540edc2997752272888e3986d9",
}

RELAY_157_WAVE15_SOURCES = {
    **RELAY_157_WAVE14_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "13f2e5a3d98793cdb4e4f5e85c56a469bfd987afaafb78308f898001b43f74e3",
}

RELAY_157_WAVE16_SOURCES = {
    **RELAY_157_WAVE15_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "ce918b309327b89bc5bafa23f19faec38f0e22a5e03ecc7337f951f83f53afca",
}

RELAY_157_WAVE17_SOURCES = {
    **RELAY_157_WAVE16_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "256d896282bdffba2f660a22c130e1d51a095ef7d5abc55bc732c5c1fa26a392",
}

RELAY_157_WAVE18_SOURCES = {
    **RELAY_157_WAVE17_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "1a1bb19270cf07dcce84b0480ca54dbc305b1f7ee1aade65ff8651683baadc39",
}

RELAY_157_WAVE19_SOURCES = {
    **RELAY_157_WAVE18_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py":
        "5ea4b2415fc2779471990e5192d9ef9a3be1f5233b20b42f651c034f6d493774",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "27ccc91f7c7d0ecff0391d1ba334209fab3fff45afd2e36450724d0fb8cfe465",
}

RELAY_157_WAVE20_SOURCES = {
    **RELAY_157_WAVE19_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "6313bcba3bb1f4c670de2d7a87f4f3e108a1effde061464f3a28abec7ccd0587",
}

RELAY_157_WAVE21_SOURCES = {
    **RELAY_157_WAVE20_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "83c7734035b8388835ef808bc23a614738555aa622443687a529566eeff2088d",
}

RELAY_157_WAVE22_SOURCES = {
    **RELAY_157_WAVE21_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "2cfcc1f180669c759553a5bc6e19ff45acce8cf4b1bad310f26a5c8ff9bc9f44",
}

RELAY_157_WAVE23_SOURCES = {
    **RELAY_157_WAVE22_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "96c180da856589073172073a6a7daf7114208a57f9cb735559d7f7739781bea8",
}

RELAY_157_WAVE24_SOURCES = {
    **RELAY_157_WAVE23_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "b47b9faaa87732c715ff25fd5512eeea157a7f45c4d7a4a7113b06571b128dbd",
}

RELAY_157_WAVE25_SOURCES = {
    **RELAY_157_WAVE24_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "d47c84bca53cd2391e9dc6554e7c48de1ebedcc9892895ff7e1c76cf976a413b",
}

RELAY_157_WAVE26_SOURCES = {
    **RELAY_157_WAVE25_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "605d0b531cd7ce4fe2db82e7545864b7cfcbba2725aca4668be19193a5014f12",
}

RELAY_157_WAVE27_SOURCES = {
    **RELAY_157_WAVE26_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "2316e50724276d1da7526b6309579a66fa491735d4e7cd2b501551f875fb22d9",
}

RELAY_157_WAVE28_SOURCES = {
    **RELAY_157_WAVE27_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "7448dd7b8e0e6da3c182fcd7c36908f21907220521e4b25523769e8d615fffa6",
}

RELAY_157_WAVE29_SOURCES = {
    **RELAY_157_WAVE28_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "ad4bee5bd220ee5fdc69ac9b595de54c49402643d257b841737b0fd2b464be2c",
}

RELAY_157_WAVE30_SOURCES = {
    **RELAY_157_WAVE29_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "901546cfc25555fef30e3fa4cd8f1b8250ad3fa3eaaf9112813d39b0481d0e74",
}

RELAY_157_WAVE31_SOURCES = {
    **RELAY_157_WAVE30_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "2f2131625931ebee929dbfd5887a06ba5d82dd9027ab531f5c7df02326f65895",
}
RELAY_157_WAVE12_PROOF = {
    **RELAY_157_WAVE11_PROOF,
    "previous_request_id": "fe395a7de89bfbf4eb8d3518e1a11e50ac02d0373c15a81f8e543245229e64a5",
    "previous_terminal_file_sha256": "6ab68e428ac4326ddaf6a9a8a73df763cffc6b7fd5395ff769968c1f4b37f717",
    "previous_transport_progress_file_sha256": "93bbdbbea43e96442938cf26d2ff1d4c0d52a17fe6624251e4b8ef24398a7267",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795713971.6597245,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE11_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "1c721d7485b5425aa43fa84b40cb906c",
    "job_sha256": "bceccf52e34a0d9c921ad46307ebd2414e613bbed7bdc080485ea614501930ea",
    "result_sha256": "2cf9c48c9fc755ce6bfd996d05e15a1b9dda4fbcf7b8f8f3d0fe34436a355199",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_157_WAVE12_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta",
    "sse_event_count": 3544,
    "last_sse_seconds": 3500.5466470001265,
    "elapsed_seconds": 3600.0235151669476,
    "provider_request_id": None,
    "model": "gpt-6.2-astra",
    "response_id": "resp_07cb3bb3604ee842016b085cedae5087d2b7ede07e8464d479",
}

RELAY_157_WAVE13_CONTRACT = "1bd3a790d40f1d8b382c4d9544e78ae5c4046f287dffff9ea897754328335297"
RELAY_157_WAVE13_AUTH = "de7355f6653bfbdbb504cc3d96fb641f60082f5059a6094ba1055f3538b7c9f1"
RELAY_157_WAVE13_PROOF = {
    **RELAY_157_WAVE12_PROOF,
    "previous_request_id": "65e609464f666e9b1e5e498195a8435755e1f90f8e5c6fa85cefbbfaea093283",
    "previous_terminal_file_sha256": "e07fe0e864788b7c4608b565757dadc78e94f475f7ccd3e2d3b5dd1e54307807",
    "previous_transport_progress_file_sha256": "687b243fe19d362494bd44777f70f2a4fdc58b30fdfac13701e753361bb2d4ba",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795717169.7967703,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE12_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "8973b62b1ef34dbc92b9140dff4757cd",
        "job_sha256": "59b123e6667bb8e011b68958b8bd47f867c9000b6e77472f4e9d42454594534b",
        "result_sha256": "8e2a0488a295e9261e37937a87551a0e779f21f06284e97edf094a1dfedf77bc",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "3ea1f797475745898d5e73835cdf80ab",
        "job_sha256": "ada5ab05c0e72c67ad17410aac2c4c92a77fd317d8fed908453d108f49cde755",
        "result_sha256": "74835ac5a00770ee44d84e33440c5c6bf81dc24304193097a6de580c0d9edb90",
        "error": "timeout",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE12_TERMINALS = (
    "3803111604c2bdd209b2ce65188aa0a3437045ef22f36da61596854ad229889b",
    "e07fe0e864788b7c4608b565757dadc78e94f475f7ccd3e2d3b5dd1e54307807",
)
RELAY_157_WAVE12_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "8ca191ce30982e122f4c55c903dd40f7574d3c98c9d9dff03f35ca063e7177eb",
    "attempt_00.replay_authorization.json": "bbe0bcf9ca35132f2caa1af53b00212771775b6686833ab855f60a9a1e20e5ab",
    "attempt_01.replay_link.json": "1b9229d17f2971f125b0063a2401d60700412466fac065ccf2300dafd1d8b5c6",
}
RELAY_157_WAVE13_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.done",
    "sse_event_count": 24,
    "last_sse_seconds": 698.6975491670892,
    "last_sse_activity_seconds": 698.6975491670892,
    "elapsed_seconds": 1598.7060428331606,
    "provider_request_id": "a1c30542-f1fc-45a6-8cae-bf0e234b81ba",
    "model": "gpt-6.2-astra",
    "response_id": "resp_090fb9d352760183016b086f72a42487d2808f1febffe6f78c",
}

RELAY_157_WAVE14_CONTRACT = "0f15c53f5fb8be71c7a92536de9afb1e5cbf90775550c876fdd93b3d216716d7"
RELAY_157_WAVE14_AUTH = "4841cbe651fd6bb8299eed6c02b6c2007fb4cc9c099e6b01b96e15b1951b2db2"
RELAY_157_WAVE14_PROOF = {
    **RELAY_157_WAVE13_PROOF,
    "previous_request_id": "ff136e4d10960d07d35821f64ef2fefe0a13779f5e8879ebbe5bb26a6e74ee9c",
    "previous_terminal_file_sha256": "9fb2c7073ef9dc8adaf022561df003cfed381025a2c698c147b68d6ffc4905bb",
    "previous_transport_progress_file_sha256": "8ad5796e24d840b54fdfb10f4d1ec200a7caa4a7cebba49006a97982b5c0b0f4",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795718562.9140031,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE13_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "2dec2b706caa412cb1251aff0a2b0b42",
        "job_sha256": "80ef8ee32467461876c7f0dc29cb9d5e25605d3c4d64d0e9a44673d38294d2aa",
        "result_sha256": "8c21b0c5c3737de3702ff38c83b1b9fb8e74c49e34c8de62d6cad6ab1ad26ddf",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "74e79fbb04454079a41244ae4ee35c53",
        "job_sha256": "ad4fc6b9129272d2e5133885b4d51de64e3f83c31d3cbdbd270e5169f2c151bd",
        "result_sha256": "16da9655cf1985c1d2c98698df2b5d5aaaa323e80b50e02c56f41b7bdcb33dd3",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE13_TERMINALS = (
    "f371380bea7f1aee557960d2f95ebd8a9f3cf1dc667c3ecaf1d169d185fab537",
    "9fb2c7073ef9dc8adaf022561df003cfed381025a2c698c147b68d6ffc4905bb",
)
RELAY_157_WAVE13_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "5a64314230eed776c69072a22a345b80bd696bcbb809dd8f78a59219486092fd",
    "attempt_00.replay_authorization.json": "71386e5b9d504db42ae5597e75b8ee747a0cb3215376977cf7ec77aba3e74a8d",
    "attempt_01.replay_link.json": "09fa2525b10dfb27e2a0cf712dae256571fb33389c7c7a5d405b8d6649adf4ae",
}
RELAY_157_WAVE14_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "provider_request_id": "5f2aa5a4-9d8a-423b-b466-bc86827d489b",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0995a2322c47b270016b087c1c135087d28df33a3dc759e82b",
    "conflicting_response_id": "resp_5f2aa5a49d8a423bb466bc86827d489b",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
}

RELAY_157_WAVE15_CONTRACT = "f0ca4d3bc4342d6b4887238415123d113545734117a03c30c51482cfc3bfb301"
RELAY_157_WAVE15_AUTH = "e8df9f783658106ad197dbeeb2d5a34f1d6e43296167aa558ffe9856dc53e47a"
RELAY_157_WAVE15_PROOF = {
    **RELAY_157_WAVE14_PROOF,
    "previous_request_id": "24c686ea0fdf4d99a23996fedd965dcc5f8f6e9120ec619c9e27a2107241f053",
    "previous_terminal_file_sha256": "96240c037c9f6683c7a041ca92cce890dd30b1629049101f6b30b442feb02adb",
    "previous_transport_progress_file_sha256": "7859aae069be83625ecddc707d2bb28a7fb87c0063cede76f16185ab91adc3f2",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795719546.705628,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE14_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "2b15775e3fbf432bac822bf518b3f6fd",
        "job_sha256": "f11b0e51e5ef96cc00907c98f3842ff65bd08191f54cb9538f06aa17d956dca7",
        "result_sha256": "2af5a898d006fbafcf2372e4ad8cf10249c0005fa980b4fffe2cb80c1ca9ad9d",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "140178c833074f75986a63a927d8ca00",
        "job_sha256": "7af1838da572e2ce29e3fb8a0d512969560de0691f729fda00febf11720a9f0d",
        "result_sha256": "37da7320abe13cd61b2fee37b218b0a6804fdcf58355b56999f897400283f370",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE14_TERMINALS = (
    "c973eff85c8b6fe49414450082face5f86c9cf15db46f0aa357287d3705cc1eb",
    "96240c037c9f6683c7a041ca92cce890dd30b1629049101f6b30b442feb02adb",
)
RELAY_157_WAVE14_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "a4d027b79027202d75008b27e4d5076b8b43e120fb0e8a4bbc9adf61a969dd61",
    "attempt_00.replay_authorization.json": "08f0528839a68bed3293dfd5ee8410dc17b14786890ce10a69a61252c5f6bcd7",
    "attempt_01.replay_link.json": "0cfdf2f1371926546b8237e098d632858467ee60faa46ac68e194f933448a9bb",
}
RELAY_157_WAVE15_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "20ce98dc-d807-4b77-a529-9fd5cf12d408",
    "response_id": "resp_00cbeebad1f62198016b087ff7e7f887d2a4d4e69b792c7d34",
    "conflicting_response_id": "resp_20ce98dcd8074b77a5299fd5cf12d408",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.38259412511252,
    "elapsed_seconds": 198.8618995831348,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.934528667014092,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.93730491702445,
                          "interval_seconds": 0.002776250010356307},
 "response.output_item.added": {"elapsed_seconds": 18.939459042157978,
                                "interval_seconds": 0.0021541251335293055},
 "response.output_item.done": {"elapsed_seconds": 28.093643208034337,
                               "interval_seconds": 9.154184165876359}},
}

RELAY_157_WAVE16_CONTRACT = "60ef6daf955bb7b11b8390cdeb08fb13db769ce47176a1ca9c09f3ec2d52bad6"
RELAY_157_WAVE16_AUTH = "c69d3995ca87f89c466f71d79c23300f302028446a93d1c2f6605df63d409e8c"
RELAY_157_WAVE16_PROOF = {
    **RELAY_157_WAVE15_PROOF,
    "previous_request_id": "bde99de6dc28ffb35e0d5083936a6c49d733fc0333fcb9f2d27127abe7403e2f",
    "previous_terminal_file_sha256": "c4fd4fabef6dfd1c48ce3e3c796622d0cf327c98f4bcada47ba64f9258192b4e",
    "previous_transport_progress_file_sha256": "9bcc2b90c83b0c8187d7a45beeaed71ce755b610b3bca414677d58741d61d02a",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795721741.325949,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE15_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "86e32af54c0e486285bb634fb55cbef8",
        "job_sha256": "39c86ffb72f1af94555e7fe79b856b0c67dcfc4348b87cd7ffc7169c106e9dd2",
        "result_sha256": "673f0600aafa83a0350589bb84d9ec3e3c35f593f98b0224d4430e43de8a848a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "0753108b819c4adcbcb54938c6d5bdda",
        "job_sha256": "9be554f96e916da6cb3235968b30febc51e0035b71a58b15538da30795972a61",
        "result_sha256": "8c9494df914662c33dab8866e1750166ce3e0885baecace9bd516f2bed7660fd",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE15_TERMINALS = (
    "812fe89ce0153b3965857e329590ab95a05d667eec8ac22479697a91d216fbf1",
    "c4fd4fabef6dfd1c48ce3e3c796622d0cf327c98f4bcada47ba64f9258192b4e",
)
RELAY_157_WAVE15_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "6bbd71fc06ebcb41679a8d421c52895c2d0d6fd4caf972ce8c52fca96790a839",
    "attempt_00.replay_authorization.json": "66a0b7411dd935b383e0d55a3b7852c432b8dffdf7635663abf62c0d07e18a90",
    "attempt_01.replay_link.json": "16fd0eed58cd80c74f3ce6578ad23d32777d362465cf9da56e727b9587c27ac4",
}
RELAY_157_WAVE16_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "44585c25-2dca-40d7-accc-ce2a1147bcf2",
    "response_id": "resp_037ec881b43e4a94016b08887ce2c887d09f04149b6aa43962",
    "conflicting_response_id": "resp_44585c252dca40d7acccce2a1147bcf2",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 210.7801252498757,
    "elapsed_seconds": 213.25315304100513,
    "sse_milestones": {"response.created": {"elapsed_seconds": 33.2462880408857, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 33.249730583047494,
                          "interval_seconds": 0.0034425421617925167},
 "response.output_item.added": {"elapsed_seconds": 33.25367245799862,
                                "interval_seconds": 0.003941874951124191},
 "response.output_item.done": {"elapsed_seconds": 49.5187750000041,
                               "interval_seconds": 16.26510254200548}},
}

RELAY_157_WAVE17_CONTRACT = "a3eff4a581df80f6fede6fb3e4961ce775744457f01598247e9849ce3eba4ab2"
RELAY_157_WAVE17_AUTH = "1d2a3b2b9c0aef5b8b704ba8fc42b395dbf510d82c4023359526a5cb79f90417"
RELAY_157_WAVE17_PROOF = {
    **RELAY_157_WAVE16_PROOF,
    "previous_request_id": "83794c225bc9400d78e15ddbd0a6891ee8688f270fc0ad0b6be55cf6f5862bed",
    "previous_terminal_file_sha256": "6bde11c304287cefe9b956f7deb61e6372aeaab1e4f8f70fcdcabc2273067766",
    "previous_transport_progress_file_sha256": "d3f46271df0f3e915ddab5bb8c2ef797cb5fafb1f3f10d650eb203179c6dee31",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795722782.7178502,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE16_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "a428b0b0658443ef9b4d1e3de274a8ef",
        "job_sha256": "4a359bee4ec1b1d62cac5fad95f928dd67d0349d886762a71a251e8095331ec3",
        "result_sha256": "4062f102d1575e541bc7bdbff43615828176f868143e75b7335d495638784d32",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "39039b9e9017416984944f65a4eb37a8",
        "job_sha256": "e0fbe6cab9255cf8a2afe3edab074d65398277331cef9d3f6de6da78dbb63c51",
        "result_sha256": "5f7ad7609a3d333f8001d62d75dfb353d2bf0711e4bda8d31f63c5a6982ce917",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE16_TERMINALS = (
    "d8d89a2ad5994beae4c1ebc18a5486a8096c8953cb4ed7e57380bafe6f0b75cb",
    "6bde11c304287cefe9b956f7deb61e6372aeaab1e4f8f70fcdcabc2273067766",
)
RELAY_157_WAVE16_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "606e09d1c2170c9878111680594d91db4477657930f591898bbf0e3f5b314fe1",
    "attempt_00.replay_authorization.json": "e2110504777b56aebf1394ed95df2f4c5fedc8c2a157cbad1c6cba78f4d18d3c",
    "attempt_01.replay_link.json": "4ba4adfeb0abce35c6c304c78251ab76dd23d28cc831f29f60f6358ea66ffbad",
}
RELAY_157_WAVE17_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "6358dd8f-9cf1-40a4-a7fa-fdd375c9b4a0",
    "response_id": "resp_0b10fe3feb21d73b016b088c9905c887d1b7ed2b30e6e4abbe",
    "conflicting_response_id": "resp_6358dd8f9cf140a4a7fafdd375c9b4a0",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 199.81623791716993,
    "elapsed_seconds": 202.5548870000057,
    "sse_milestones": {"response.created": {"elapsed_seconds": 22.720137584023178,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 22.74440075014718,
                          "interval_seconds": 0.024263166124001145},
 "response.output_item.added": {"elapsed_seconds": 22.77850366709754,
                                "interval_seconds": 0.03410291695035994},
 "response.output_item.done": {"elapsed_seconds": 31.997200459009036,
                               "interval_seconds": 9.218696791911498}},
}

RELAY_157_WAVE18_CONTRACT = "e8c52f6f5e746c73b56b9e34e674f390b435146ed6f300805b47c954fae186ba"
RELAY_157_WAVE18_AUTH = "ff6fe7f2950033d6cb21f34d36e6a70c5db658f2f57694bcea029db6af1fbadf"
RELAY_157_WAVE18_PROOF = {
    **RELAY_157_WAVE17_PROOF,
    "previous_request_id": "be139e4d191fe3f8feae14eae42e4df765baf4c3d57cd33b5ed0b1f362a89006",
    "previous_terminal_file_sha256": "8927aac5d427f40752ae4d3c50da49309df7317448d2292c8c20b92d859ae786",
    "previous_transport_progress_file_sha256": "78f37204a861717cf0dfac6027c7e43870d6acba946b0bea057b87b7039bdaa3",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795723851.023255,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE17_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "21368d1641eb45258b3c573d8e05c6c8",
        "job_sha256": "dfd2f7d6b28c1d225e3601b81dd0c9b861b7a1dbfd60f83825ee1bf1ab7aaf5d",
        "result_sha256": "06a08942620bd2e3e8097478bba5b326d0b7caa285b1c910310bda5e17d01e96",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "9e607209d29c4dc0a5553f3f3cce9942",
        "job_sha256": "8f3a2a85edbca768c6880ff692c3a033c7a05595f64b91299829d23093b41565",
        "result_sha256": "7d18c1e6f509cbea40d1f7299686d9167830ee9926d1278002a63a4a9614e645",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE17_TERMINALS = (
    "f085c4c7f362da41c9ecc9683153707ef4a7b6ecc3f488a1d2363d2f39364378",
    "8927aac5d427f40752ae4d3c50da49309df7317448d2292c8c20b92d859ae786",
)
RELAY_157_WAVE17_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "7a626bedb178f43eaf0d43eccb083f514baf8bc3ec53b11363cb3833143152fa",
    "attempt_00.replay_authorization.json": "094722a58836331606ed36181fd430f2cb2c573820a8c94b8ae872dc19d84686",
    "attempt_01.replay_link.json": "8d2a0d9d90accd864ad831053362c0d2541642d2c20090b6b0ab2678bc136ca9",
}
RELAY_157_WAVE18_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 39,
    "model": "gpt-6.2-astra",
    "provider_request_id": "8bf07202-460b-47e7-a0e1-760d769b2c16",
    "response_id": "resp_0ef06a18ce3cbe09016b0890cabf0c87d1b9ff4b2e5b8cef53",
    "conflicting_response_id": "resp_8bf07202460b47e7a0e1760d769b2c16",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 199.53460570890456,
    "elapsed_seconds": 206.96631662501022,
    "sse_milestones": {"response.created": {"elapsed_seconds": 31.603634458966553,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 31.605146833928302,
                          "interval_seconds": 0.0015123749617487192},
 "response.output_item.added": {"elapsed_seconds": 31.611024416983128,
                                "interval_seconds": 0.005877583054825664},
 "response.output_item.done": {"elapsed_seconds": 44.093582833884284,
                               "interval_seconds": 12.482558416901156}},
}

RELAY_157_WAVE19_CONTRACT = "d613f006d5563188c15a76e345a902e7b9c9a8cc134aea74fc25760fa8aab862"
RELAY_157_WAVE19_AUTH = "24bc96d658f867936a1f306317e09429af05552af5f6ed787694554faad9b854"
RELAY_157_WAVE19_PROOF = {
    **RELAY_157_WAVE18_PROOF,
    "previous_request_id": "6e0635eb183a9921de26ff86560f2ad11d0dde26b82ee5315b31e8f911d5801b",
    "previous_terminal_file_sha256": "616c2e1b13491406c543ddb968d66cb3f3dd919b9ed0ab51d802d0482d29adde",
    "previous_transport_progress_file_sha256": "84bbab0a142c49200987787113fce71461ad57c6d750840dd96fbd4bea1c6416",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795725903.7298713,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE18_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "dc7cfa5063e54558beb40e16fba7a0cc",
        "job_sha256": "ae3ab8a0daa8f3849f9ebc77d761083c98566a5dcf8b66a8f930fe078ea08b7f",
        "result_sha256": "c721ef963f44c6139f7212d423d561d3dd53bec84915cb3709103f1cdf820b64",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "6e2250d2243c49b3888b1c92dd4ebcb1",
        "job_sha256": "a0e850ac8eecc4059420a6344d009a83c443b5a17b381893a46e4932885b9732",
        "result_sha256": "b0bbe0a3a2cc2a876ec12c221b00fbd11283fac27bf2c4ff53d3916095857779",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE18_TERMINALS = (
    "5a1c97ba41c1dd3c3c5df2a699e53c8059264e70f5b4e3bf4dde0dcd209f731c",
    "616c2e1b13491406c543ddb968d66cb3f3dd919b9ed0ab51d802d0482d29adde",
)
RELAY_157_WAVE18_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "46fe156552c86721fafd3db97c63372916e4a7c6bec6ea17700813808de3f8e7",
    "attempt_00.replay_authorization.json": "44adc2554e9507e69a247c98de94a2cd785573fad4b6354c5f2ab214c473dae7",
    "attempt_01.replay_link.json": "eff15bd38ffae5997f2f4bfbdc129854ec5a5d679df2fd7ee0e2075838ffdf41",
}
RELAY_157_WAVE19_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b76a116d-7b48-42da-b76a-b2c515091c21",
    "response_id": "resp_06ef494d4a709fd9016b0898ce801887d2bdafdc5109af59b9",
    "conflicting_response_id": "resp_b76a116d7b4842dab76ab2c515091c21",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.2681829999201,
    "elapsed_seconds": 200.00862212502398,
    "sse_milestones": {"response.created": {"elapsed_seconds": 20.034967333078384,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 20.03709787502885,
                          "interval_seconds": 0.0021305419504642487},
 "response.output_item.added": {"elapsed_seconds": 20.03945575002581,
                                "interval_seconds": 0.002357874996960163},
 "response.output_item.done": {"elapsed_seconds": 29.1976025828626,
                               "interval_seconds": 9.158146832836792}},
}

RELAY_157_WAVE20_CONTRACT = "9935b79b98c7f107beee8dd9ec1d90d95e778411099c869367dba3b8d2792ee9"
RELAY_157_WAVE20_AUTH = "45733b57666b64bc529abbdb6f6b40b4a38c147ae36e419d0d4510d9767b5f6c"
RELAY_157_WAVE20_PROOF = {
    **RELAY_157_WAVE19_PROOF,
    "previous_request_id": "172ebd981ada49931004781d55dfb28c6ebba1543c062b617962310c48f2e7ec",
    "previous_terminal_file_sha256": "313bf93d351ae6ca5ed2fb177c76b517aef5ae9f02ae245b6851c790c9d8029a",
    "previous_transport_progress_file_sha256": "e2259b99a395b4fce239ddc98ab11c1c264078539ea168381efaa3c3b45f9399",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795727243.4760141,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE19_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "3eee29e09a504e3ca05604684fd42780",
        "job_sha256": "833fd0b9125ea36eea71d760036b033301011d9f4d86d7a105e8a3ec236bcffd",
        "result_sha256": "1821aadcaf1498e1d192a09374af9e16d84b53f7a1a4e3a747f09d0a9aceb74d",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "c027d0bb376640a59a9127deff102363",
        "job_sha256": "102c350d59fea52cf9f18eada25f6e93ab77c89bdd96df0683406d970cc8ff4d",
        "result_sha256": "caf621298a8eab94cfb378d9e29baeb741c318f8820c4c6e573077f053e8375c",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE19_TERMINALS = (
    "4c58a5ef02f9d710f4f28e53f2bf214876a8dc1b6f52bef665487df5ce1c121e",
    "313bf93d351ae6ca5ed2fb177c76b517aef5ae9f02ae245b6851c790c9d8029a",
)
RELAY_157_WAVE19_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "4074fb1342c26de8c1eb15743970d4a9610f501eb23799bc738380a4391b07b4",
    "attempt_00.replay_authorization.json": "4749661d7fca63e1c4d427e2d221547b43a735d5828334dace6f8a9ec4558f06",
    "attempt_01.replay_link.json": "7c09a96596a2f6027f5b1c37ed004804e4afe90cb9f91cf065bf15e86993cd4c",
}
RELAY_157_WAVE20_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "e9e45cf3-5ee8-45e3-adea-de3b115d824d",
    "response_id": "resp_05f7313513895fa0016b089dfeab4487d186f98ab482f5d3d4",
    "conflicting_response_id": "resp_e9e45cf35ee845e3adeade3b115d824d",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 207.11027549998835,
    "last_sse_activity_seconds": 207.11027549998835,
    "elapsed_seconds": 209.26974008302204,
    "sse_milestones": {"response.created": {"elapsed_seconds": 29.327437083004043,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 29.329424374969676,
                          "interval_seconds": 0.0019872919656336308},
 "response.output_item.added": {"elapsed_seconds": 29.332867791876197,
                                "interval_seconds": 0.0034434169065207243},
 "response.output_item.done": {"elapsed_seconds": 47.64363854192197,
                               "interval_seconds": 18.310770750045776}},
    "network_phases": {"direct_ipv4_http11": {"duration_seconds": 29.3200689998921,
                        "observed_seconds": 29.323291207896546,
                        "state": "complete"},
 "direct_ipv4_preflight": {"duration_seconds": 1.0522868339903653,
                           "observed_seconds": 1.057941332925111,
                           "state": "complete"},
 "direct_ipv4_preflight_1": {"duration_seconds": 1.0522868339903653,
                             "observed_seconds": 1.0556331248953938,
                             "state": "complete"},
 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.18832250009290874,
                                     "observed_seconds": 0.19192058290354908,
                                     "state": "complete"},
 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.1945246250834316,
                                 "observed_seconds": 0.38822079193778336,
                                 "state": "complete"},
 "receive_headers": {"duration_seconds": 25.0867345421575,
                     "observed_seconds": 29.318083750084043,
                     "state": "complete"},
 "send_body": {"duration_seconds": 0.7547209588810802,
               "observed_seconds": 4.228899291949347,
               "state": "complete"},
 "send_headers": {"duration_seconds": 0.0001393752172589302,
                  "observed_seconds": 3.4739071670919657,
                  "state": "complete"}},
    "first_phase_seconds": {"http_headers": 29.324781416915357,
 "http_phase": 0.19192058290354908,
 "request_dispatched": 3.471844542073086,
 "sse_event": 29.327437083004043,
 "transport_failure": 209.26974008302204},
}

RELAY_157_WAVE21_CONTRACT = "0e2c3407f84bb056870db36755a830924586ef467107e729a0e61cdf87abe81b"
RELAY_157_WAVE21_AUTH = "16fc32b62b823f3ce82de86055ef803d08c4c6d4c4b25d60cb42ba777f4ebede"
RELAY_157_WAVE21_PROOF = {
    **RELAY_157_WAVE20_PROOF,
    "previous_request_id": "23a23024f1f3a784e66302e5e055fb64d6e854cb8c5f8c4df3bcc4e002f41679",
    "previous_terminal_file_sha256": "b1c6f17416f5ca38fc74e34747afdd360366becf7c838a23cb582262a5a08e9b",
    "previous_transport_progress_file_sha256": "ff27b14a8e2d06c20ab40929a88bee4226c731b14b69b1570467ebe9868578ad",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795728380.9793127,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE20_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "aa56b198c9b04413ace9ca158b0fc34b",
        "job_sha256": "247833a35e6cd63cff4e50fb0dab6ea1d666378563aee4d65744c6863143cf8d",
        "result_sha256": "2a8c59d26dc03d24d2bd473b90255d4ca8e97cb4140a53ca9bc1c9e9ad34365b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "7a9c1ecb0d024f189769ef2d2954374b",
        "job_sha256": "86a84ff9706c6b23ae2949ea5066c586878aa62762ec6c92f5f6a9b25348f5f3",
        "result_sha256": "6c1854ee27ef0ab3b1d558f579c8393f25c008e3dccc0de5a4bec8f692b1bf77",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE20_TERMINALS = (
    "9082931239072337a8e46317dc0c8ae1cb19b47613cfbf12d62ffbfcc90ec047",
    "b1c6f17416f5ca38fc74e34747afdd360366becf7c838a23cb582262a5a08e9b",
)
RELAY_157_WAVE20_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "74d3c428e925e351ffed88eeeb7578f3bbb9b3dcf20638064e025cfefe135338",
    "attempt_00.replay_authorization.json": "0d6a128e4b2a0e9dafede3d861318d517818f7c505b7f5bd4e4eb464cbd43cf4",
    "attempt_01.replay_link.json": "62712afb21826d24a80ff2cc5aa9e6f61188eb4f44cd7e6169cca2ddbd7a0b43",
}
RELAY_157_WAVE21_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 25,
    "model": "gpt-6.2-astra",
    "provider_request_id": "fc9191d2-7cb7-49a6-9ca9-5cad7e93d45c",
    "response_id": "resp_0bcbdb82d1ac7eb2016b08a274c11887d083d338f413c3a9a1",
    "conflicting_response_id": "resp_fc9191d27cb749a69ca95cad7e93d45c",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 205.73171916720457,
    "last_sse_activity_seconds": 205.73171916720457,
    "elapsed_seconds": 207.95196458417922,
    "sse_milestones": {"response.created": {"elapsed_seconds": 27.94564904202707, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 27.94786670920439,
                          "interval_seconds": 0.0022176671773195267},
 "response.output_item.added": {"elapsed_seconds": 27.949778834125027,
                                "interval_seconds": 0.0019121249206364155},
 "response.output_item.done": {"elapsed_seconds": 44.31273345905356,
                               "interval_seconds": 16.362954624928534}},
}

RELAY_157_WAVE22_CONTRACT = "752e6a56d1796ed6817b05da2afc873efc4c0b6db404c7565905c2d1ad9b43d6"
RELAY_157_WAVE22_AUTH = "7e7684b8033561fa040381c883074e09b31525e69fd795a3d97f0b7150ef2873"
RELAY_157_WAVE22_PROOF = {
    **RELAY_157_WAVE21_PROOF,
    "previous_request_id": "f0504b4a24091d4da009bbbb3a38ab3e4894ada1e1a08ea8fcbfe95104e62c29",
    "previous_terminal_file_sha256": "3f5029b698dbbac214f7826a53111deb066b980c791abfbdcac9d0a87cc0fc00",
    "previous_transport_progress_file_sha256": "1290f9fc10cbe9ce3c036c2e1b57ab8dc42ce6fda84c330ea8d123cc62893473",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795729527.1188662,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE21_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "b41e1069da06490dba37e4cf783a19d5",
        "job_sha256": "95e5b2a5e685451399da4eb1a2e27f3e96daf15acd58b409703c07fc71e885c1",
        "result_sha256": "35d81f4b88e226e6184041c92f96ab7738c6ac202523d26b83a8d60ffc9c6fd5",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "4bc6d117ceb145dba341f1e694f903e8",
        "job_sha256": "2dd2c0c083dce95aff243c6883e8eaeb15b30dcb5db1489a20adfc61473b218f",
        "result_sha256": "10d1482c1c46879f132f4edb8a5a1f54f30a6fce1a064e771e4861c88a532d4b",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE21_TERMINALS = (
    "47f6405a933f740def543e2948d852b08eb1a7a1ae718a19051c8d60a730c5af",
    "3f5029b698dbbac214f7826a53111deb066b980c791abfbdcac9d0a87cc0fc00",
)
RELAY_157_WAVE21_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "184f09c5cd9ec56827e5dc39db0bb282149f7da4ed43639ac3f225871a6d32e0",
    "attempt_00.replay_authorization.json": "5286bf394052d7ba82a7b9504cf98d693831d4dec9c1152d6d8bac064cdce869",
    "attempt_01.replay_link.json": "6622c6a7d32e4b70efc6e25120a9ba721668f2a3e9cb629af7fc9292a0d4f07b",
}
RELAY_157_WAVE22_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b8951463-48bd-49d5-8022-f46189ddccb4",
    "response_id": "resp_0a2662c58705c65e016b08a6ef67a887d2ab7e6f3a1d86c0c0",
    "conflicting_response_id": "resp_b895146348bd49d58022f46189ddccb4",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 194.418092665961,
    "last_sse_activity_seconds": 194.418092665961,
    "elapsed_seconds": 197.21956591587514,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.275104415835813,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.27725474978797,
                          "interval_seconds": 0.0021503339521586895},
 "response.output_item.added": {"elapsed_seconds": 17.280243249842897,
                                "interval_seconds": 0.00298850005492568},
 "response.output_item.done": {"elapsed_seconds": 26.639862957876176,
                               "interval_seconds": 9.359619708033279}},
}

RELAY_157_WAVE23_CONTRACT = "0e8357f1ce54a69ed9f98d462fa9bff9cfd2a902b1ab419e934251fb9235f72e"
RELAY_157_WAVE23_AUTH = "35c74688d1ee0a102bdb11098b5bbbfaa80d941ffbab8c6b764289bdee83e647"
RELAY_157_WAVE23_PROOF = {
    **RELAY_157_WAVE22_PROOF,
    "previous_request_id": "b10df8e26573ee3dce765103d9c1a455cfbd554ca5b1055be4cb50abe8d460dc",
    "previous_terminal_file_sha256": "33b6bf126eba81acccbce695d4e0de5ae1da57612e1ccbf8dbdf48c6af3a6432",
    "previous_transport_progress_file_sha256": "7579d68bf8edf91209f4f7d88188ce9cde91ec6a306ce129aa4ddd258df9f480",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795730515.952652,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE22_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "2b9b653c5a98441eb5d17613841cbcc6",
        "job_sha256": "3465b4a67d97669c197a2bfbc167f366f0370306327178f5b1a2cbf8fc0eb448",
        "result_sha256": "24abd4bb288190d75117686c1e70fa00c0e422e2e60c8700802cc4f27287a50d",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "33fabf638c754f0ca1655aeb45b3f763",
        "job_sha256": "0b99590ab8b4e2af8663bd7507316eb402dfe66d2622315b179f3fba5db6f6a6",
        "result_sha256": "c876dcb80c7ab5946c4e0f453cc5dbd3db309e69a6fb1890d92b56cddd5b7eae",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE22_TERMINALS = (
    "17d93807928b22536fc71f443bb5833048a55597cde2ca74391a2aade757d744",
    "33b6bf126eba81acccbce695d4e0de5ae1da57612e1ccbf8dbdf48c6af3a6432",
)
RELAY_157_WAVE22_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "a28dd9a9f3404204650fa22e31ce59359a2b2635fe1236ec4a9f43cccaa6a90e",
    "attempt_00.replay_authorization.json": "e891e2aa0d289c9c1b7999a748a5d01a68de9ea5d57b62654fa99a4e1668ea4e",
    "attempt_01.replay_link.json": "1bbe0332724f4ee700bf577ab568840e926e3d2a5a379628437e466df2c2d6f1",
}
RELAY_157_WAVE23_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "0c304b28-2b77-4c4d-a33f-3742db1d64ce",
    "response_id": "resp_09a0a8dd60f15e58016b08aad27b5087d29915b446ce8d5a9b",
    "conflicting_response_id": "resp_0c304b282b774c4da33f3742db1d64ce",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.4820779999718,
    "last_sse_activity_seconds": 195.4820779999718,
    "elapsed_seconds": 198.22378837503493,
    "sse_milestones": {"response.created": {"elapsed_seconds": 18.278119042050093,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 18.279653459088877,
                          "interval_seconds": 0.001534417038783431},
 "response.output_item.added": {"elapsed_seconds": 18.28185429191217,
                                "interval_seconds": 0.002200832823291421},
 "response.output_item.done": {"elapsed_seconds": 27.594514833996072,
                               "interval_seconds": 9.312660542083904}},
}

RELAY_157_WAVE24_CONTRACT = "774a6760ad71cbdcc51fe976b9f5a0674f4631350aae5ba89302dfb8aa9deb56"
RELAY_157_WAVE24_AUTH = "e1070df1052b5c98a76db3f9e5c738b5f244aaff7703062e2a1bd54f70fa422d"
RELAY_157_WAVE24_PROOF = {
    **RELAY_157_WAVE23_PROOF,
    "previous_request_id": "bd9d1fa27271276b7e3f0f7972e301c64bae808a521ecc9a16eb5b0cd04964d2",
    "previous_terminal_file_sha256": "5a1c837c7398939e407815fee96cfedbd7e158400ff9fa3d466e8c5f772fb32c",
    "previous_transport_progress_file_sha256": "bfe944c6c4c0bd43e8f30b44443bb09ce31597e58799104dd4c1ba887ae57dca",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795731435.068868,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE23_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "afbcb8adb16847d097eb676509fe245a",
        "job_sha256": "b317bec12e06552acfc750df7d4f16793c5b0df6eb0963a0bb0dfad37b15f215",
        "result_sha256": "d6e980d10c178e73249a77a271049227f2791241c809765316c15d17a55f5d05",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "56dcb2c4d8384847872b8cda68e4de43",
        "job_sha256": "5903bac054ceb9405f7e8aee0de110c5116cfdc891807bfdcb0cd40531906baf",
        "result_sha256": "6a5aa138d0781a9ed8c98ae7019fbb9e2a65390d56498b4cca10dfa02d0251d6",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE23_TERMINALS = (
    "83eccebb686f2e0a10d24956edf72a1c2a8caf2c61930c8c886b5ae037193309",
    "5a1c837c7398939e407815fee96cfedbd7e158400ff9fa3d466e8c5f772fb32c",
)
RELAY_157_WAVE23_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "629c73e30e5ac97720e0fc0bfda9b03d1af11e1d2a166688ed51a4078a859bba",
    "attempt_00.replay_authorization.json": "5b5516ae557ff1cbb8f3baed23945892c80208f7475148bf9649c086b19c2e43",
    "attempt_01.replay_link.json": "de2327c88b26f99a6baaad0da64ae73bd974a62adccd4adf4b9f6967447d5e93",
}
RELAY_157_WAVE24_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "4a62968f-fc78-44ce-96f3-26e9af0fc8aa",
    "response_id": "resp_0c5cd60d3e13f521016b08ae6cdea487d083ae473d6ffacb66",
    "conflicting_response_id": "resp_4a62968ffc7844ce96f326e9af0fc8aa",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.19101420906372,
    "last_sse_activity_seconds": 196.19101420906372,
    "elapsed_seconds": 199.0167657500133,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.083091042004526,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.086583208991215,
                          "interval_seconds": 0.0034921669866889715},
 "response.output_item.added": {"elapsed_seconds": 19.088602042058483,
                                "interval_seconds": 0.002018833067268133},
 "response.output_item.done": {"elapsed_seconds": 28.337056583957747,
                               "interval_seconds": 9.248454541899264}},
}

RELAY_157_WAVE25_CONTRACT = "2b092bbbfd85540c4501cb49ff4879592c598c883d6b858d3b0a909419e0ac47"
RELAY_157_WAVE25_AUTH = "6c097642f58b7837b35fc27d119097b3301ebb24ff57d59d474bd8f7f7cb87be"
RELAY_157_WAVE25_PROOF = {
    **RELAY_157_WAVE24_PROOF,
    "previous_request_id": "1b1c7ccdce821d3d1c56a3c3a248a364104b355d39ec49c8d16a531f8d65d25b",
    "previous_terminal_file_sha256": "e6bc3ce452e40ac7a2ee1204b44fd3743850b9e83524a64ce16e9c7a99c12cb5",
    "previous_transport_progress_file_sha256": "461b9f6d0414d6e60643f0c3d05530498dc1fcd5e71f52f98df38963675e9470",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795732390.0013366,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE24_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "d4ddddd9d12a4410bc9da955510b954b",
        "job_sha256": "cd4f539d9922c9b3e8ae65350f3bb57f40bd4253fb57c8b7439c35177bd81e04",
        "result_sha256": "7378570a90cbd878b3fb9818834102f3b9a2a5f4ef753e898b47915dc896f39f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "3f49545be09a48caaa7e58eff1e59183",
        "job_sha256": "2a7ea476a80378ecbe6f3b193107e5efecd142bdfb47d8647ba225efb93ba4c8",
        "result_sha256": "817389dbf0f7e676477589ad4bdea4529143ff7e4b17405095d60c4a6c3dcd81",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE24_TERMINALS = (
    "591591be3f162772f976b62ea842fc53d4c440d570f234f35dfbb663168541ac",
    "e6bc3ce452e40ac7a2ee1204b44fd3743850b9e83524a64ce16e9c7a99c12cb5",
)
RELAY_157_WAVE24_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "0ecf3156a18ec5ababe8e7e71bd807a2b3c1935e78f7909c9d5250662c5f268a",
    "attempt_00.replay_authorization.json": "10045d5cc2a8cdfce2361d8fa4fe884f41e05c84637cebc3cd5650e513b0a81d",
    "attempt_01.replay_link.json": "ef2f001fc3323256e498535f8ee4c1f34d5df06618430052c7e16b3475100f6c",
}
RELAY_157_WAVE25_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b62af458-d456-4e26-ac90-60af4f7e6cfa",
    "response_id": "resp_034482b19d562975016b08b221dc9487d285057aae13818770",
    "conflicting_response_id": "resp_b62af458d4564e26ac9060af4f7e6cfa",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 194.74844729201868,
    "last_sse_activity_seconds": 194.74844729201868,
    "elapsed_seconds": 197.48023062502034,
    "sse_milestones": {"response.created": {"elapsed_seconds": 17.54852154199034, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 17.550381666980684,
                          "interval_seconds": 0.0018601249903440475},
 "response.output_item.added": {"elapsed_seconds": 17.553042375016958,
                                "interval_seconds": 0.002660708036273718},
 "response.output_item.done": {"elapsed_seconds": 26.910217207856476,
                               "interval_seconds": 9.357174832839519}},
}

RELAY_157_WAVE26_CONTRACT = "c3128811ae5674d9092b9317b65cf7e73abf02b6a612548924bf44bda9c13d5e"
RELAY_157_WAVE26_AUTH = "a1a5928a02db08b957e10c6ff8410bd585d24c1e3bd5d7a4399c89066d1ae69a"
RELAY_157_WAVE26_PROOF = {
    **RELAY_157_WAVE25_PROOF,
    "previous_request_id": "2938f4ad574e8f41741a72dadbde88c4a25b2cf13fcb88a75a506fa05e14ab94",
    "previous_terminal_file_sha256": "1f6bcf4541f48b16b36d7956c6f6386126fddb43a9884e1d3da19a6ed00b7830",
    "previous_transport_progress_file_sha256": "a178623f423dd7c69b56e21fca37e5dbaae46ba1dc8e52576dd765871d23144a",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795733574.8626847,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE25_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "efa33350473941da9ea4e7db87d73d11",
        "job_sha256": "9ac719e5d594c720f0d43003229e39bf4ea73c5f0b60441cd6b9bab80c1612bd",
        "result_sha256": "02ab63094b740ad3bd236cbb61720e3c654f02d9f5d7a885d8c77894ab7703b2",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "85ab8b7135e742a3abaa5f67101bc5d2",
        "job_sha256": "36775bac7ad28c6ff2dd41b23106bc42ad5268c9c244b55bf6169edd2d0b6b21",
        "result_sha256": "2d7dc159eb63b31eb09d278d48e72cf1434cff1d4a19e426b4c1657be5bf4e55",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE25_TERMINALS = (
    "69f6dce3bddb2e3b2ad0237edf164e60bb64c733abaddf34cc790fa45972bdce",
    "1f6bcf4541f48b16b36d7956c6f6386126fddb43a9884e1d3da19a6ed00b7830",
)
RELAY_157_WAVE25_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "76fddcecf20a9b1c59e37be4471eec91a38e0ded9616850aae99962a88a88594",
    "attempt_00.replay_authorization.json": "0583c5b95478424ddd9a581353311839ce7157cc4d8164a5c70d55680cad9feb",
    "attempt_01.replay_link.json": "c6440a0686d16c0db3969760ff1a137ce480f37e2a7269b7390b44f241954c39",
}
RELAY_157_WAVE26_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "c26c286f-caf0-4606-8d2b-b56c1f98f7e5",
    "response_id": "resp_0ab23737dd4240c7016b08b6c10fd487d0989a2f3718daebce",
    "conflicting_response_id": "resp_c26c286fcaf046068d2bb56c1f98f7e5",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.2643029158935,
    "last_sse_activity_seconds": 197.2643029158935,
    "elapsed_seconds": 200.13134933286346,
    "sse_milestones": {"response.created": {"elapsed_seconds": 20.153173875063658,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 20.155819833045825,
                          "interval_seconds": 0.0026459579821676016},
 "response.output_item.added": {"elapsed_seconds": 20.157977750059217,
                                "interval_seconds": 0.0021579170133918524},
 "response.output_item.done": {"elapsed_seconds": 29.418403457850218,
                               "interval_seconds": 9.260425707791}},
}

RELAY_157_WAVE27_CONTRACT = "1ed4fc579298721c3c254ec3c213ebc508ca6acd843038a5d521109a68f36bae"
RELAY_157_WAVE27_AUTH = "4e32e30893900aae039a4cac8dc038791cff52fb78eeba6ce46f8c1acc51585b"
RELAY_157_WAVE27_PROOF = {
    **RELAY_157_WAVE26_PROOF,
    "previous_request_id": "c8b31c12b1e010e044849ca4d663a61cfba008bdf37d263849c36dc4edd13bff",
    "previous_terminal_file_sha256": "b6688422ffbc826efd1a1657036a7eb7cad37bee7e79baa5dc868693c8d41d3c",
    "previous_transport_progress_file_sha256": "ed9c2d42b0e98bd95f34d1bf16267660a76e0a60384c3000615dd71f1a85c7e8",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795734523.5643582,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE26_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "e2c4454234bb4fbf98c2bd9997833edf",
        "job_sha256": "56d75fdb3942baea8de7f278ae31796137769042ce21bd23e468b449a346fed9",
        "result_sha256": "224179710c99520514a977cb14678849d4f75b9c8c5a6dd03be75c98c4112a9a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "3571d6d09f704437a838c8bf2024f709",
        "job_sha256": "312fd71fda63247abbd1e6125308b3eb4efb6e615e9284964f2c0f7ab41da902",
        "result_sha256": "f4e5fea219205604456fd18f36839e1af927597dd080505e782e1d064f87ed16",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE26_TERMINALS = (
    "773dba409e12ae2f9b9f02af0e37d8eae620b685fc19c7846d3628cfb0ad381c",
    "b6688422ffbc826efd1a1657036a7eb7cad37bee7e79baa5dc868693c8d41d3c",
)
RELAY_157_WAVE26_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "5419d40ffdf083d07899d031bb21b49d170b835cc63ec73896ea88c6af1723a9",
    "attempt_00.replay_authorization.json": "1a9a7f8c14382f364f5f174dfdaff25fe1f2ea76f09ddb36ca58195812da9824",
    "attempt_01.replay_link.json": "cff7a4ae6948a098cce7d86f35028e1bb3c9140a4bf2909baba1062ffa1062d4",
}
RELAY_157_WAVE27_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "59c31069-5005-4b56-bbd7-a88a7ad76918",
    "response_id": "resp_04128c057b9bcb48016b08ba77bbdc87d2908aa834fdebdd21",
    "conflicting_response_id": "resp_59c3106950054b56bbd7a88a7ad76918",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.09684495883994,
    "last_sse_activity_seconds": 197.09684495883994,
    "elapsed_seconds": 199.73100983398035,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.91749866702594, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.919983208877966,
                          "interval_seconds": 0.002484541852027178},
 "response.output_item.added": {"elapsed_seconds": 19.922454458894208,
                                "interval_seconds": 0.0024712500162422657},
 "response.output_item.done": {"elapsed_seconds": 29.07027466688305,
                               "interval_seconds": 9.147820207988843}},
}

RELAY_157_WAVE28_CONTRACT = "484fd934c622ab7d0ce3c8107b976c7827362f94574a690d4955a98851445181"
RELAY_157_WAVE28_AUTH = "7f83161dc79b4437bf1c2a49a6d9e7fa36b1fa231f9b770abd91ee473e5d75b5"
RELAY_157_WAVE28_PROOF = {
    **RELAY_157_WAVE27_PROOF,
    "previous_request_id": "5738e3a3aeb3d046e33cc948dd6d0a418ee635d9f3754c26ed9225ab0d8b955e",
    "previous_terminal_file_sha256": "b2e1dd5d66d860d7d4085cb3f36600070c86b7a0046c88c2416a2e5bcbeacbf6",
    "previous_transport_progress_file_sha256": "cddeacafc28f2248749bfc6510c818ace876af776b5835e0f16bffde6f496c99",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795735518.7383046,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE27_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "00718d2c5eff4ebd91cb3057f1698b14",
        "job_sha256": "25798e1b46c5422785ac3eacdcc478d9503fd8915703e40366830f591bd2e3be",
        "result_sha256": "5dded59584ee9a9f3040f70d62f04c9f3b7bafe7041036a695263ccfd849252d",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "4dab6aef3a0d463b8e479c56bde37874",
        "job_sha256": "1dda3eac85628cb232551a18418f0512c43c77520203cc61c6364712a9611f63",
        "result_sha256": "031e5634d81e1f258f56a1c36bdc20a73315937a4693d73536dd3cc0b14954fe",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE27_TERMINALS = (
    "4ea0599c0bcf2d2eec16bbde7b9662ed23d24504d5bcd4a580b8064a8b2a9f6b",
    "b2e1dd5d66d860d7d4085cb3f36600070c86b7a0046c88c2416a2e5bcbeacbf6",
)
RELAY_157_WAVE27_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "c10ead5296c63fcf0396eed370de97a479dcc65c84f43e966ed04216c2cc9b85",
    "attempt_00.replay_authorization.json": "9f9bb8d3b8156891cee2493fb7ff3384c35693b98e36abed1e707f49db3ae579",
    "attempt_01.replay_link.json": "40a5f180ecbd82e547129aa276474ac3fb65ccfc1d5ecd44a54c635f815348a5",
}
RELAY_157_WAVE28_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "efd9b982-8cfb-4665-bed0-0e15ecd511fb",
    "response_id": "resp_0a8aa2239525677f016b08be597d8487d0a5c52f326caa09ed",
    "conflicting_response_id": "resp_efd9b9828cfb4665bed00e15ecd511fb",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 195.87420912482776,
    "last_sse_activity_seconds": 195.87420912482776,
    "elapsed_seconds": 199.01104666595347,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.022447832860053,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.02447075000964,
                          "interval_seconds": 0.0020229171495884657},
 "response.output_item.added": {"elapsed_seconds": 19.027008374920115,
                                "interval_seconds": 0.0025376249104738235},
 "response.output_item.done": {"elapsed_seconds": 27.93340133293532,
                               "interval_seconds": 8.906392958015203}},
}

RELAY_157_WAVE29_CONTRACT = "9e06289040f1b241e198249a4218930b042e7fa95b299c227d9ab5d6fda2e766"
RELAY_157_WAVE29_AUTH = "2fc829c35b9e0c325d35c23400b0e454ae845d165e4ac9ce960ad2f8888625b7"
RELAY_157_WAVE29_PROOF = {
    **RELAY_157_WAVE28_PROOF,
    "previous_request_id": "e5befd4afcc9636fffebd3aa0b75ce68478b550ed100000c3d555a4da00d4d8b",
    "previous_terminal_file_sha256": "c29d5feb98d2a687b810c7095f170703cbd311836d4c65b97c192b8a49548727",
    "previous_transport_progress_file_sha256": "a2a111d776c6c3066c6ce79dbc90cfef204b97ffb21505e47f7c1796faba6da3",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795736565.0167,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE28_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "3c03b8c5136c457a8b56706c46e48481",
        "job_sha256": "2cc5f73078fb7bc9f3aca493772c27b6edc768502d43cef0b6bfc3e3aa836b20",
        "result_sha256": "4a1e3466f8fa19a93e34cb9e40860a6e63cba9cc9c188f1925792cd55a6b8a30",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "c202018f7191414880907775b4ec2479",
        "job_sha256": "0d61755dd8eb336805f3b6fb808337d1f0cb648ae0427489c0509b5e2480bbca",
        "result_sha256": "af01a6f8052247d075d0c1b2570b8d3c49a118ee06ce5da6501a6e4b9f5bdb78",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE28_TERMINALS = (
    "ce671501df2d3688b771f89f9e7a3e4af096ffa70e7518ab04af094515c0e8b2",
    "c29d5feb98d2a687b810c7095f170703cbd311836d4c65b97c192b8a49548727",
)
RELAY_157_WAVE28_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "7b0b287430e79765489078e22fd1ffa2ca886ab9039e12b2dab5ba8a588a1805",
    "attempt_00.replay_authorization.json": "3322d2569c606b4cadd99f530b1b29864d1dd5529697c37ecd27b27d7ea73974",
    "attempt_01.replay_link.json": "0a24c498909e27ec85d8bc22821902e85a16e9c50f41214f01f1e6ba7997658c",
}
RELAY_157_WAVE29_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "77703397-692f-4daa-a99d-4b53ae24ee86",
    "response_id": "resp_0c50575277d71e5d016b08c275b23c87d0a355f12b9ee105b3",
    "conflicting_response_id": "resp_77703397692f4daaa99d4b53ae24ee86",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 196.74496558401734,
    "last_sse_activity_seconds": 196.74496558401734,
    "elapsed_seconds": 199.37997475010343,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.40484083397314, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.406724834116176,
                          "interval_seconds": 0.0018840001430362463},
 "response.output_item.added": {"elapsed_seconds": 19.408693167148158,
                                "interval_seconds": 0.0019683330319821835},
 "response.output_item.done": {"elapsed_seconds": 28.718067083973438,
                               "interval_seconds": 9.30937391682528}},
}

RELAY_157_WAVE30_CONTRACT = "7a61efd03fb204b02eddd74b5b7fb3777b98faed8bafa31ec81881fcb6759129"
RELAY_157_WAVE30_AUTH = "8074700f3d2e1a211a91445f3d29fc3c9350aa54f3c264788851fbdf2d8d2cb2"
RELAY_157_WAVE30_PROOF = {
    **RELAY_157_WAVE29_PROOF,
    "previous_request_id": "f20484681207f76c36d840f3b13afe0e2924f022fef9561a119d8abcff180629",
    "previous_terminal_file_sha256": "642eba1fdd04da762dc73c29c8ae087471b344f1e9aa93768b94df9309426dee",
    "previous_transport_progress_file_sha256": "47a06d4693aec060a15736a133d9ca61bd652b4cc879c74831349cf0f9890095",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795738643.1450398,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE29_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "777d3b522a6447c0b6565c5f62c9762b",
        "job_sha256": "50b952c754d3e08d5ee04c2b07e49959193c6a2d11cb542c3b5752b8adc08f57",
        "result_sha256": "fcf07d984f2ebaed2c18ef1cbab8b7fa05190b90720c0207803f1eec48457a6f",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "53a1f934df2a401695fb4ea12e24f6cb",
        "job_sha256": "e7802e2a233a9651961ebb629f1c6d4d37ae08a8017c34b28f1556087684bcc0",
        "result_sha256": "68a8e33e400f327328d4fab0d143bc7f5234129cb2cfcc8c426b936fa2f0371a",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE29_TERMINALS = (
    "ab143c4679e721dd1b9d923eb657070f4a1f16bac76058ceff54094cc455ee73",
    "642eba1fdd04da762dc73c29c8ae087471b344f1e9aa93768b94df9309426dee",
)
RELAY_157_WAVE29_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "f3fb5895784032286bbe252033d436e2619dddeadb395ebc8e9f26476a30d1d7",
    "attempt_00.replay_authorization.json": "e059903aa8e06a1c48597c96cf27bbb9487b3bf53d1ad0508b56666cc86d57ae",
    "attempt_01.replay_link.json": "c3754ac833a4dd7d49bfd2b16b511a57426160c8d7a6b1a7aaf8aa16249e222f",
}
RELAY_157_WAVE30_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "b4f10d7c-dd63-46fa-8660-a3f7391f05e7",
    "response_id": "resp_02e319e4edb7216c016b08c6a6c44887d1ab7398095d8f29e1",
    "conflicting_response_id": "resp_b4f10d7cdd6346fa8660a3f7391f05e7",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.01793545787223,
    "last_sse_activity_seconds": 197.01793545787223,
    "elapsed_seconds": 199.61267120786943,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.689311124850065,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.693044916028157,
                          "interval_seconds": 0.0037337911780923605},
 "response.output_item.added": {"elapsed_seconds": 19.695783957839012,
                                "interval_seconds": 0.0027390418108552694},
 "response.output_item.done": {"elapsed_seconds": 28.981517457868904,
                               "interval_seconds": 9.285733500029892}},
}

RELAY_157_WAVE31_CONTRACT = "7dc4fe8710d9e92d74856e3275b11ed5527867561e7c1fd0b5fa4620ebe363e0"
RELAY_157_WAVE31_AUTH = "e2b90738fca6ab45ca2fe9ff00822d8c0255a0a2d9ba3e418b3b28075f35d524"
RELAY_157_WAVE31_PROOF = {
    **RELAY_157_WAVE30_PROOF,
    "previous_request_id": "f2613a1e76afd37a516a9861c2f342f03460212e897032fb8e4a3a5af0f79313",
    "previous_terminal_file_sha256": "c2ea039e916c8b3a1f8475896a3d0d9286808ab400a9f0e307e982df7443f1f7",
    "previous_transport_progress_file_sha256": "0b827be19b69ee4de6dbbbd5b051611995998be101b9e53a1c562f54b340cbde",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795740329.9381309,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170},
}
RELAY_157_WAVE30_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "1f856802fce4415a8bf718fbe73645dd",
        "job_sha256": "86720a11dc23b17d646b0450df76f81b489c0d0ba2341c39b6082785662eb04b",
        "result_sha256": "1a30be4e11b0b48e0e7a4a8e5c07d9d3e775fab9f93feeeccf1ceb214a10bd98",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "0bb8a2c34e2c41d0a7d64c1b4474e37a",
        "job_sha256": "b3564d29e937689d7adcc947cbaa2ce5301db128e6b35a650185bdac7d7c1f6d",
        "result_sha256": "fd8016e457bf41a7c2f7b84e5afb3f81f561ac9f88b6f3b03798f65c3098fa81",
        "error": "sse_frame_error",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_157_WAVE30_TERMINALS = (
    "ea260a269df18fdcb524587bef5c2b704c771f0f2f246e6745f0bd88a3061966",
    "c2ea039e916c8b3a1f8475896a3d0d9286808ab400a9f0e307e982df7443f1f7",
)
RELAY_157_WAVE30_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "9af8f8ec1b636573162ea7fa530ed2cdd391c69f578ac4b3391798fa80570640",
    "attempt_00.replay_authorization.json": "bfe3e5cdc631c574ef9b5e849b88ce250af20053f591e0aacb56afa74a388e71",
    "attempt_01.replay_link.json": "d64f88de82b3f573de4a41c0a5b5c641aa61a2fa72a78749e6e68889424cc3b1",
}
RELAY_157_WAVE31_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "ReadError",
    "event_type": "response.output_item.added",
    "sse_event_count": 33,
    "model": "gpt-6.2-astra",
    "provider_request_id": "03d70d7f-0219-43df-831d-c32464ef6e9a",
    "response_id": "resp_03d7479f6dc20303016b08cf7a714c87d1890afbeec1e1c39a",
    "last_sse_seconds": 584.0116467080079,
    "last_sse_activity_seconds": 584.0116467080079,
    "elapsed_seconds": 629.9220020000357,
    "sse_milestones": {"response.created": {"elapsed_seconds": 20.187189458170906,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 20.189439125126228,
                          "interval_seconds": 0.002249666955322027},
 "response.output_item.added": {"elapsed_seconds": 27.404897791100666,
                                "interval_seconds": 7.215458665974438},
 "response.output_item.done": {"elapsed_seconds": 32.87080904119648,
                               "interval_seconds": 5.4659112500958145}},
}
RELAY_157_WAVE31_ARCHIVED_SSE_COUNTS = (41, 33)
RELAY_157_WAVE30_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE29_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE28_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE27_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE26_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE25_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE24_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE23_ARCHIVED_SSE_COUNTS = (27, 41)
RELAY_157_WAVE22_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_157_WAVE21_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_157_WAVE20_ARCHIVED_SSE_COUNTS = (25, 25)
RELAY_157_WAVE19_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_157_WAVE18_ARCHIVED_SSE_COUNTS = (25, 39)
RELAY_157_WAVE17_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_157_WAVE16_ARCHIVED_SSE_COUNTS = (41, 25)
RELAY_157_WAVE15_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_157_WAVE14_ARCHIVED_SSE_COUNTS = (25, 41)
RELAY_157_WAVE13_ARCHIVED_SSE_COUNTS = (41, 24)
RELAY_157_WAVE13_EARLIER_ERROR = "upstream_error"
REPAIR_887_PARENT = "dd0501bb9680a2af013c4a5b22976b64ff1b7d691b2645e0b9dab0191564de65"
REPAIR_887_CONTRACT = "a176ab59cc1c069ac77f67dc15516844bae3d3f6fa13fde6e7a77db73afa2247"
REPAIR_887_AUTH = "666fca40036ab340100c819bb0ccd58fec9239aab6b2ee792f6ffcf8e2b98c92"
REPAIR_887_RUN = f"{SERIAL_IPV4}-887-r8-targeted-repair-01"
CACHED_887_R8_RUN = f"{SERIAL_IPV4}-887-r8-cached-correction-01"
CACHED_887_R8_CONTRACT = "edf9e9d68b2d47924962bad4f7b3f277f85702d00cee52cfc1a2ac204790da43"
CACHED_887_R8_AUTH = "ccb3d9b7759efcdf1911327cf35a52fe01518057146f33e037b18a6a992c8801"
CACHED_887_R8_RECEIPT = "7cdbe9a47be2448b50f099afccab7804cac75c3361e7cd45c3f99f58f31129b1"
CACHED_887_R8_PROJECTION = "401c78f2d7317fa6ae7dc0473382ecf7880753a2dd805fc73936188516516860"
CACHED_887_R8_BOUNDARY = {
    "checkpoint_sha256": "5618dce31ecc72da5cbc337f734ed3cae50075baa1d2750ff6e77e00e0287c07",
    "extension_source_sha256": "0b33cbe7d235940cd51ea5f0a934ec34805c9ce7b10e7c6bc9844b1daadd1cbd",
    "extension_transition_sha256": "e9dcf1774bd75453c10fd483bf953dab2cb1e08b43c10ff974b9b0a2b4b1c2c1",
}
CONTINUATION_887_R9_RUN = f"{SERIAL_IPV4}-887-r9-continuation-01"
CONTINUATION_887_R9_CONTRACT = "58de5fc9e6a56844cc720404a3ceba8f8e7c8445c4f25290f175748a9d0c9d4f"
CONTINUATION_887_R9_AUTH = "d32b845eb0e8cbfbab28c523f82acba853c1e8df345e492033c2458a879658cc"
CONTINUATION_887_R9_BUDGET = {"first_logical_requests": 27, "maximum_logical_requests": 79, "maximum_physical_attempts": 158}
RELAY_887_WAVE05_CONTRACT = "56dadfb27df633863b7a8e31ce103446fba7ea2a3b0f13188ffea5a3f7af5daf"
RELAY_887_WAVE05_AUTH = "d40358da6676fd0ce3d75ae507d6c3abe8443124ac8772479766ed2feb20b604"
RELAY_887_R12_CHECKPOINT = "68f0414b95ed886540b989083c5e1c58bc96304fb3178b117d8cde85e8182603"
RELAY_887_R12_FITTED_FILES = {
    "boundary.json": "b68021e483cd762af998caaeac7a5f03ff151a8f7331213a851765f2c25a67c0",
    "round_09/source.json": "0b33cbe7d235940cd51ea5f0a934ec34805c9ce7b10e7c6bc9844b1daadd1cbd",
    "round_09/boundary_memory.json": "f7dc25bf194a2538139d1caa566df9a64ff98a967d268132152c74782e55c2c6",
    "round_09/memory.json": "a18263a6de3410b4b1841e861b8d836c9dbc9e29d123c1f7cf6c379eee65337b",
    "round_10/transition.json": "8608338a1c622370eca3ff3692e79c7ed8ebd57d003f701525cc4adfd8d596e6",
    "round_10/memory.json": "46196709e12b669d0ea5894ce50d41e3273e7e4b41eedc63f1577db2f7294638",
    "round_11/transition.json": "08f35a25536ac2a450eb7faa827f8c385ac8d1555c7c44a89b9fb4dacb097ca4",
    "round_11/memory.json": "1175f594a4c9dc046a50858f2353ef990e42062771d7bf20fe601e7487683d0f",
    "round_12/source.json": "fefcfca59556ff2a8623feacbcc5dda078a44f05ad006d9c5af712fa1117e3bb",
    "round_12/transition.json": "8bae92f10d48972c566ad18a4fd868c0ab904447daad26222401ab75f6664b05",
    "round_12/fit_000/fit_audit.json": "76d60c67bbe4034975ec2b7c250d4ee7b7767c3527859a3505f9cd8babf5eb3a",
}
RELAY_887_WAVE05_PROOF = {
    "logical_index": 0, "payload_sha256": "4064b0a2694a7d36ece124937bbc5c47b2c5e4e77b22b8384a2af998c15c5b38",
    "previous_request_id": "a53987367a8eddc7f4b0bf731e290388b42e1ade9b0c5c3416a5344f5021b2f7",
    "previous_terminal_file_sha256": "8cde8a42b2fbfbb4c3602142add3f0ebfcdda7e58f8e3b47c53f90a8502f0ce9",
    "previous_transport_progress_file_sha256": "9c84d32abf0ca52029e6cb144a47255577d57e99984ea33698faa83674c79c76",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795697557.9016292 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 139},
}
RELAY_887_R12_ATTEMPT = {
    "attempt": 0, "attempt_id": "75647f5eaad9426d8f7cc13ebfa490b8",
    "job_sha256": "80e6838f4e2bcfbc61ebb99a80e4712f982fef48cad8abe06ee1ea33d33ef7d8",
    "result_sha256": "49fb3999e75a5ccc343a10ecfef40c7b796f971847b111f282cf2b1e3f771514",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_887_WAVE05_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_text.delta", "sse_event_count": 207, "last_sse_seconds": 961.4464893341064,
    "provider_request_id": None, "model": "gpt-6.2-astra",
    "response_id": "resp_0aceb6badc01f076016b08246221d087d2a30ef3e67e38261f",
}
RELAY_887_WAVE06_CONTRACT = "05df8c27277d8ef241e0e24eebee334b7e261f93a26f8687279135a0c795e7a6"
RELAY_887_WAVE06_AUTH = "7c7b2c6a3b6c521568df409f0813b6378b3c72cee701f5972ef368e12383eb8e"
RELAY_887_WAVE05_ATTEMPT = {
    "attempt": 0, "attempt_id": "f0fcbe7e61894a389e1f6f2715b0081c",
    "job_sha256": "0c497d13b0c326e29b2e94dd6387308e2f4d7752ff979d8424d3b193a71e407b",
    "result_sha256": "53db427b46fc66113949696a70c99b7377e1d49857df219c02d7b0ede4cad5ec",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_887_WAVE06_PROOF = {
    **RELAY_887_WAVE05_PROOF,
    "previous_request_id": "8eaac87cccc02053f168a8d32860d34ab8af33291832dd148e1a9bac0c9c8d4d",
    "previous_terminal_file_sha256": "f7cec4bdff72d7fc9fd789aa4ed117d066cb0accd86ec3386151fa85b1fd1cc7",
    "previous_transport_progress_file_sha256": "323deb41f369ff271b089ae7f894e7a9dac58f085bce7874326ad7ce5f447e43",
    "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 1795701794.7327738 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 140},
}
RELAY_887_WAVE06_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta", "sse_event_count": 3063, "last_sse_seconds": 3017.041881625075,
    "provider_request_id": None, "model": "gpt-6.2-astra",
    "response_id": "resp_0f3cc35efc34af43016b082e2f53ac87d29e05725c60e2bff5",
}
RELAY_887_WAVE07_CONTRACT = "e1521cd0a4df8d040e307e72f2d446e0f5020368227c9d0ef7a0f0c4122abc94"
RELAY_887_WAVE07_AUTH = "be8e4fd97acea0f28408a47f405dd2631e67a862c0294f6b5ed6d5272c747a68"
RELAY_887_WAVE06_ATTEMPT = {
    "attempt": 0, "attempt_id": "bd3433a1d2924c038d11fcc28c9c50ec",
    "job_sha256": "18c75f6cb93d97bbe50e24bf3845ad2c6a9b6a2d34aacc22f17da95e586c948a",
    "result_sha256": "c3bfb08f768c2b36e9954cedb14c0ba83b7cd603cab002749665f97e2fdbb1bb",
    "error": "timeout", "provider_dispatched": True, "ambiguous": True,
}
RELAY_887_WAVE07_PROOF = {
    **RELAY_887_WAVE06_PROOF,
    "previous_request_id": "a21b75b60154249e0a6f115269ec581e595cf674249f3993672a02a4bdc126ba",
    "previous_terminal_file_sha256": "b411f9727ce4d310961bb7acfbdabe0a5681aaa25a52b2cb71bfddbf5ad768c4",
    "previous_transport_progress_file_sha256": "d4db95abc9c0f37a85e94b7830c0920f74003169974372121c3cef4e2ee41b07",
    "relay_cooldown_clock": "failed_unix", "relay_not_before_unix": 1795706314.0892487 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 139},
}
RELAY_887_WAVE07_OBSERVATION = {
    "http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
    "event_type": "response.output_item.done", "sse_event_count": 4060, "last_sse_seconds": 2876.7962896670215,
    "provider_request_id": None, "model": "gpt-6.2-astra",
    "response_id": "resp_0ed690650e4d4829016b083fd8387087d0ade644c2fa35138a",
}
CONTINUATION_887_R9_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r9_continuation.py":
        "f10d2266cee09aaa848069e3b9760b2298419799ed5ade066ef116104dcdf4c6",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_887_r9_continuation.py":
        "e5174b01905bc0ca36a7cb19b142ba683aab937a80e18e19fbfe79af424a213b",
}
PILOT_ROUNDS = 10
TARGET_ROUNDS = 36
TARGET_LABELS = 1485
ORIGINAL = "a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10"
FIRST_887 = "a7-gpt62-astra-xhigh-ssh-v5-prompt-pilot-887"
PILOT = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
SEEDS = (157, 887, 2357, 6101, 12203)
RELAY_887_WAVE08_CONTRACT = "dc53baf5ce4f7aa74bdfa300d38bee182735132ddac9294eab940f07ecd7157b"
RELAY_887_WAVE08_AUTH = "dcf42f11bf0a7294b2b7e0e61dc5c2f27e03d3be1f26b1f379de5df68067a5e6"
RELAY_887_WAVE08_SOURCES = {
    **RELAY_12203_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "31293c5db211f4970f4bb61ce7265d8283ecf8a3ef8c8472e4b4c13757b325de",
}
RELAY_887_WAVE08_PROOF = {
    **RELAY_887_WAVE07_PROOF,
    "previous_request_id": "0b7052c4d92bb87d7043d1b974da626a35330ce0f44ea3466b5f44ea07181f0c",
    "previous_terminal_file_sha256": "c509d8b291b11b5951409ffebe50f67d03b9635cb4aafc2bad71df617bbd45f2",
    "previous_transport_progress_file_sha256": "b2adc0ee127762c715d227dd7676190be9e8c1c07f2f20d8aab8882c6b78c4d5",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795709072.5213063,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 140},
}
RELAY_887_WAVE07_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "d820f33e0ffc4a96aec41a0bf9ef6afd",
    "job_sha256": "9f67b64d9165d2e19686a0e2d8ee15e72de3a9822aa559661e46c33f05b727c6",
    "result_sha256": "4433da0b23521da441117d9bb805a2126455a0d10e89747e6223a8797ec8395a",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_887_WAVE08_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_text.delta",
    "sse_event_count": 1347,
    "last_sse_seconds": 348.2114551658742,
    "provider_request_id": "175117fa-25c0-40b8-b11a-23960a32a5bd",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0c10cef05b83b84a016b085029790487d1b095a46533e7c77e",
}
RELAY_887_WAVE09_CONTRACT = "8408218609f6a2597f575872027d9063e755193e9a19ecc1b11c185596f9f9f2"
RELAY_887_WAVE09_AUTH = "e3dfe88654c69a1bee57dff0ac61e9446b4369e9063abec6bba449b6afd08cd5"
RELAY_887_WAVE09_SOURCES = {
    **RELAY_887_WAVE08_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "a549d1aa7451c317664231536de814b02c3ab54f17edf7ad395f52bc8d603f9c",
}

RELAY_887_WAVE10_SOURCES = {
    **RELAY_887_WAVE09_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "cb31983ecd8d2fff3b6491e8f4b34cf35f6d0038062e6a57c3b4311a2a7d61b6",
}
RELAY_887_WAVE09_PROOF = {
    **RELAY_887_WAVE08_PROOF,
    "previous_request_id": "2677b92a0b9886f5e295317d7e8d42188a226b7ff53554c80606f85fb8252414",
    "previous_terminal_file_sha256": "d251850e10d8e66f795b77213ae0a11943c049817000a8080d4412db7b27d3fe",
    "previous_transport_progress_file_sha256": "44140b6b9f3e0358077181667c1b462eb772c0a8be6d8fc89bca84129a8efdfd",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795714038.0125747,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 139},
}
RELAY_887_WAVE08_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "083e46db66ab4d6d862e7b21e1d270e1",
    "job_sha256": "106cc3576359c6113a6f5af8442eb097bd5a559b80b8fce330c63197eb30a6df",
    "result_sha256": "9d796f0f9727a12acf840d7d9a7e4fdcdb9113278db4b006c3c88d436625d217",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_887_WAVE09_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "hard_wall_deadline",
    "event_type": "response.output_text.delta",
    "sse_event_count": 4679,
    "last_sse_seconds": 3551.905509167118,
    "last_sse_activity_seconds": 3551.905509167118,
    "provider_request_id": "32b23c7e-dde0-42bb-92b9-f54b74090afa",
    "model": "gpt-6.2-astra",
    "response_id": "resp_091d3a6b10f6db84016b085d5f6be487d1832b50cb44fc0739",
}

RELAY_887_WAVE10_CONTRACT = "cc90be12c858a5b2a9d2e1687c0dceec5dd249d1d7fee4753c9210a9f054950d"
RELAY_887_WAVE10_AUTH = "f31d42cdfed27143ce9a42d5c72c4a4bdd054d8496f5e78bbf90f07b7b169cb2"
RELAY_887_WAVE10_PROOF = {
    **RELAY_887_WAVE09_PROOF,
    "payload_sha256": "20b9b243194cbc330493d023e3f27928fea6d7f05ae8d900c33b413a9670dcb4",
    "previous_request_id": "4cd434414bb2a004cf502820a697c108563791901d236f56da48554ed26a557e",
    "previous_terminal_file_sha256": "a7a4b3c0e234c0ba382d6d83d76d194a3df457e21cce20a7b88ae1eb0f81d707",
    "previous_transport_progress_file_sha256": "b0f4ef098235ec9888f93138edece746ed17d4df6ce277b9a72b799ccde892bd",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795717168.4392984,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133},
}
RELAY_887_WAVE09_ATTEMPT = {
    "attempt": 0,
    "attempt_id": "45aa34b1ce2e4057a50e125986247374",
    "job_sha256": "d09ef244178f2ece349133c58f0132ee079b0d9ace936a35327996bce3f92ca7",
    "result_sha256": "548edea79c0185f5a909e8360567be3eb3bbef173e228997f9375087f8558ff5",
    "error": "timeout",
    "provider_dispatched": True,
    "ambiguous": True,
}
RELAY_887_WAVE10_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "sse_inactivity_timeout",
    "event_type": "response.output_item.added",
    "sse_event_count": 33,
    "last_sse_seconds": 698.2018592918757,
    "elapsed_seconds": 1598.217064499855,
    "provider_request_id": "1d4b6c50-a782-4beb-942c-428ab95712bc",
    "model": "gpt-6.2-astra",
    "response_id": "resp_0e4cc43d31cd7c17016b086ea8394087d286ecf2b81d61e187",
}
RELAY_887_WAVE10_CHECKPOINT = "dced0207b5677e2b5b1d1937a40af0d26bae92cdfc6d72518a95410ac431a152"
RELAY_887_WAVE10_FITTED_FILES = {
    "boundary.json": "f8e16e48fcbb21596d7f953b3486e2a5639cf745195959b6da881a8f3dbc8228",
    "round_12/source.json": "fefcfca59556ff2a8623feacbcc5dda078a44f05ad006d9c5af712fa1117e3bb",
    "round_12/boundary_memory.json": "1175f594a4c9dc046a50858f2353ef990e42062771d7bf20fe601e7487683d0f",
    "round_12/boundary_user_message.json": "fe4cb1ee67adc272f372a701901238617447c19b47b28d923bb8728f5f856f6e",
    "round_12/selection.json": "bdfff944548e2c28b3f787abfcb218fdb64bc9f48d1e83f8886bd29a82794a33",
    "round_12/batch_freeze.json": "5d7ce14dff50d459e5859e30f4c94b1fecf079830cf4a07b9c17935e8431dc30",
    "round_12/memory.json": "58763db2a3995f943fb84f7f9b32c937705b4606b1c43a988079e84132380410",
    "round_12/turn_00/attempt_00.terminal.json": "9713bcb1e5fe1ef5a5ebf14864ee56f1104a6558143efe541f42de1f117c3d92",
    "round_13/source.json": "242f308cc4fa10fbe86d31bde6e349f31c5745c2a8b47a3588468c92fb008be1",
    "round_13/transition.json": "4d4f93c6958478d309ff42560626118cf5629a3aa6323a5a069bcfa734a69f33",
    "round_13/fit_000/fit_audit.json": "11b20513827e0c9bc6c094557de4f2d2e0adfa38529ca482e44bb692855791cb",
    "round_13/fit_000/validation_metrics.json": "3858914983e81f6e16302d6ed82b7cbc48133f4e673b6b78a7e746ba1027b132",
    "round_13/authoritative_packet.json": "fc15e56dd9c646aa7155986e15343c4c5548529023a8a136b9fde024106b0c9c",
}
RELAY_887_WAVE11_CONTRACT = "a25b339b3663b90ca6932d106cf03c34ef21799784a1eb303dd8fb64f6829417"
RELAY_887_WAVE11_AUTH = "7a458fa0066f4a6f47a01f26d1f1759cb14a69635e9c50de9d6a1369bef0b784"
RELAY_887_WAVE11_CHECKPOINT = "ac23374d4dd7f7276053d0d333707ccc03862aae40c7c419db01ac40806100a7"
RELAY_887_WAVE11_DESCRIPTOR = {"round": 32,
 "state_sha256": "ff365a45b3d884960b3a4e7f47801ebbd7c2a93b00d41392c15303a4ca1e8b9f",
 "memory_sha256": "bfed21d220639d097a716a41136eb4d5e355a75712fb03833138c4ac9c6ca230",
 "user_message_sha256": "cc826c5a5b2e2de31c77d5fbcc89d5f69196490fb626a3e5a368ae345974a713"}
RELAY_887_WAVE11_PROOF = {"logical_index": 0,
 "payload_sha256": "302b17f2672c1cb477425d7aaa8ec077e443c07f6b4015a4499fb54b5e5d9b65",
 "previous_request_id": "7f0007dfc6b13e98ec4cd2048924ffa5f252543cc39a53a9b6ed797a1e16beab",
 "previous_terminal_file_sha256": "9f4c44ba5702e45a78eac04a62b660fb7decc9c64e5ff89a1382d528db0cbe64",
 "previous_transport_progress_file_sha256": "201fbfcb589974fd6e1b22c507147578c0cb9f5d26a3e2e26d38a0a7cb6468c6",
 "relay_cooldown_clock": "not_before_unix",
 "relay_not_before_unix": 1795731145.8450844,
 "newly_authorized_boundary_physical_attempts": 1,
 "execution_budget": {"first_logical_requests": 4,
                      "maximum_logical_requests": 10,
                      "maximum_physical_attempts": 19}}
RELAY_887_WAVE11_PREVIOUS_ATTEMPT = {"attempt": 0,
 "attempt_id": "bede371adb36462082aac5b39355d648",
 "job_sha256": "b63239eb02e2c6d71743fc127e09f0dab6b39cd410ac291edf917cc1a8e7b91c",
 "result_sha256": "f3472191db9b5db955e7c64b4eaebf81e82bbccc9cab0f0b099bdb46f92f8389",
 "error": "response.failed",
 "provider_dispatched": True,
 "ambiguous": True}
RELAY_887_WAVE11_OBSERVATION = {"http_status": 200,
 "phase": "transport_failure",
 "error_type": "response.failed",
 "event_type": "response.failed",
 "sse_event_count": 9,
 "first_sse_seconds": 9.765055833850056,
 "last_sse_seconds": 24.46110837487504,
 "last_sse_activity_seconds": 9.773610417032614,
 "elapsed_seconds": 24.466253083897755,
 "model": "gpt-6.2-astra",
 "response_id": "resp_ft_8b4f0119e61b4184bd7b87a7dfbcda7a",
 "provider_error_code": "upstream_error",
 "provider_request_id": None,
 "sse_milestones": {"response.created": {"elapsed_seconds": 9.765055833850056, "interval_seconds": None},
                    "response.in_progress": {"elapsed_seconds": 9.767424499848858,
                                             "interval_seconds": 0.002368665998801589},
                    "response.output_item.added": {"elapsed_seconds": 9.769154584035277,
                                                   "interval_seconds": 0.0017300841864198446},
                    "response.output_text.delta": {"elapsed_seconds": 9.770635250024498,
                                                   "interval_seconds": 0.0003777500241994858},
                    "response.output_text.done": {"elapsed_seconds": 9.771671583876014,
                                                  "interval_seconds": 0.0010363338515162468},
                    "response.output_item.done": {"elapsed_seconds": 9.773610417032614,
                                                  "interval_seconds": 0.0010443332139402628},
                    "response.failed": {"elapsed_seconds": 24.46110837487504,
                                        "interval_seconds": 14.687497957842425}}}
RELAY_887_WAVE11_FITTED_FILES = {"round_20/source.json": "79f91d8382922a05f29a239176e7e9e20b48756ee3a3813e77f152cb640d3ae6",
 "round_31/memory.json": "ec18fb358a4bbd94dc384ad0671f568e9be700ee1d379f8cc16920167057ca3c",
 "round_31/batch_freeze.json": "caccafc445a769b5ec83754f4094ff63a5e80656cdeb606060bfd81d8c7e8f01",
 "round_31/selection.json": "175cc1905010f05a91b2bb946337a6355d1c91eea2a233a6bf4f73c7ed9679f3",
 "round_32/source.json": "35b8b5f750460dd39c9fd98e70b1e9ae167f25889d438d2720c7d9745aa1e9d3",
 "round_32/transition.json": "6e03687d135eb8696e1dafae86766368f34a284c85cdf4d559ff34c18ec3f7ef",
 "round_32/fit_000/fit_audit.json": "ebf3cb221b13939bf50d25bab0e0e70120f803b2fa25914aee39bb3e0a8d5a25",
 "round_32/fit_000/validation_metrics.json": "05e57e272b50bf9542d1f1a6d33440cc3da731f0fb672f106b9aed1d9a4b3047",
 "round_32/authoritative_packet.json": "8c9099c4600386baae1435633754edccf14291839d59fa252d997d1431357fda",
 "round_32/turn_00/request.json": "b067a98ce415d93ba15c4e7045083d7863a8937b1db3a9622366f64478ad2fbb"}
RELAY_887_WAVE11_SOURCES = {"studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/dispatcher.py": "cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/provider.py": "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/seed_runner.py": "19122c372294d5dd0f08d2ba4b1388b027ac1bacf65d84004c5eacf5b498e9f6",
 "studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py": "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py": "1216375461d021bfa5acf023689cb4d4b8a39df25b91e534975a645852aa0e3a",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py": "6111c04c5c1eb6688e5cf6d0b086d4c1d8179a7bfcd7a9e9d04e8f572fce38de"}

REPAIR_887_R32_RUN = f"{SERIAL_IPV4}-887-r32-targeted-repair-01"
REPAIR_887_R32_CONTRACT = "653ec5d2732e72bfc317c084c98671890030ffda0c361cd52570fd1ef22bde50"
REPAIR_887_R32_AUTH = "7a429145f5aaf60783e82fad5ca18975d3b53fd180066cd25946b8ebfe95956a"
REPAIR_887_R32_METADATA_SHA = "2e298ecf09bffafdac9f8c38417e22ec54e570c744fba3532bc49907a07a4c16"
REPAIR_887_R32_METADATA_FILE = "6fdb1f4d50d728691413c618ad3e08d5163bfd8df23dd26a88351a8ede5726af"
REPAIR_887_R32_CHECKPOINT = "98f69f161819ef20d0921b33b5fa60403de90595057535f71d9fdfb9e134a7f0"
REPAIR_887_R32_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r32_targeted_repair.py":
        "c09912fd0433c9bfcec8fb99afdf4c353153fa7eb029750393f5a06c5a9fd219",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_887_r32_targeted_repair.py":
        "15fb8912a486f30b910bed622c692a51e0850adea9897f791d3993ee74b569c9",
    "studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py":
        "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c",
}
REPAIR_887_R32_ORIGINAL = {
    "seed_887/pause.request": "a6b52c37675de268e7e7f2c4e1635d1e37d446c4a7ea29984b62e825efdc2c7a",
    "seed_887/round_32/turn_00/attempt_00.terminal.json":
        "631109e6c9adafb0521cf495836a87c5660e88466336f3854f0e55331d32e306",
    "seed_887/round_32/turn_00/decision.json":
        "72c96738f1829101c021a7ef2d6746a6a29795bcdae739cb3e336986a38897e4",
    "bridge/results/bafc2f8bfad14ca89af0b79a79e12064.json":
        "47242366342c2626881bd236e8ab4d7b06d952bfe12a07266b0a8428e2e98238",
}
REPAIR_887_R32_FITTED = {
    "round_32/batch_freeze.json": "b74f38a12bbd9e12e160d860382d12ae2cfbbf82c09249ee15380d6b6a7afb9a",
    "round_32/selection.json": "d349b9dd93d6ce398ccdcc75cf598adfc63c2ec76638b6671ba263a42ce8d44c",
    "round_32/turn_00/attempt_00.terminal.json": "7f7e9057a070f2754878c849249f804941e581818f3fc67e4337c47e3ecf8d37",
    "round_32/turn_00/decision.json": "718864eaa21eb589f971976b2500542aea2624e6283ad7e4d7eccfea5e44b8b4",
    "round_33/transition.json": "f86fa61bb92fa90c41ac29c8dc5171730996d5ac517fa4dc696cb38c5d278640",
    "round_33/source.json": "51a79ba4929ff53020dfbcc5a22b596b1369ae638ac50cf051de5314ab2a2add",
    "round_33/fit_000/fit_audit.json": "dbbf4a6bc0e3d4de2258778ac9f0538b8a56160a5a2de6d80605b92bba85d927",
    "round_33/fit_000/validation_metrics.json": "89f21681dba322d3ab68932b4b7f382522312a18bab20df73fc8dc989fd72fee",
}


CONTINUATION_887_R33_RUN = f"{SERIAL_IPV4}-887-r33-continuation-01"
CONTINUATION_887_R33_CONTRACT = "b85af24f05525a09c03617eaeca9a10d79c3fc5709726f21ef93885d42b63ec0"
CONTINUATION_887_R33_AUTH = "2efc06435a36ce310dc9bb8518a31586a219c685f9c5400a5178b9cac9440dd5"
CONTINUATION_887_R33_DESCRIPTOR = {
    "round": 33,
    "state_sha256": "dec721e5e40fd0ac25d4e21ed11a1c72f678891d9baef75c92ce9ae182c46e2f",
    "memory_sha256": "eb0af3e7de942ef0bd099a3c9597efca22a734ff5652fedb357c3b754c3886bd",
}
CONTINUATION_887_R33_HANDOFF = "63fe92b0c4923490cb2e61b68734b884752e0e419d71f198314276b1a2337ed2"
CONTINUATION_887_R33_REQUEST = "0313ddd125670786dd7f7f1b7fd5fc51fd9ac450b998299fba787a189b983913"
CONTINUATION_887_R33_BUDGET = {"first_logical_requests": 3, "maximum_logical_requests": 9, "maximum_physical_attempts": 18}
CONTINUATION_887_R33_SCOPE = {"maximum_logical_requests": 3, "maximum_physical_attempts": 2}
CONTINUATION_887_R33_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r33_continuation.py":
        "a2cc98ed9fe14c6cc9f82d73aa6f998f5ce2357fce6cee89c84f8a662de1626a",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_887_r33_continuation.py":
        "74a78b5e5764ddee327b7f2619424459699c1d5927d1ae5c25be3b0b815e5bb5",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/seed_runner.py":
        "417f41255d53d7794abcdc98c0a9258141d01c076df845bfa877013c84de1792",
    "studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py":
        "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c",
}


RELAY_887_WAVE12_CONTRACT = "d3432162c64cc96d608cddb4e9a8ac5cff66a98eb95a217fa9bc7bbf186fda91"
RELAY_887_WAVE12_AUTH = "f9eede5051316c7aa1d5dd77afcb6f63de4a89cfac63ca8ff35d8272605d3e49"
RELAY_887_WAVE12_CHECKPOINT = "c81f34b104de3873a9fd54dfad4323b118dcab6e208281d3c3e95db68c5fe710"
RELAY_887_WAVE12_DESCRIPTOR = {"round": 34,
 "state_sha256": "d26efdc5bd224cfbc958af9edeb899c6ebbd4737897b953c2cd4a44678115261",
 "memory_sha256": "4be9a81d022532472d1dca9b8f9b6b0d1ba1cb278c1d581cfc9f64c6c7111ec4",
 "user_message_sha256": "bfd87b5ce71b63c183619e5219ba02ed9d6f441d9eb6f1278a05c1e3deb7df19"}
RELAY_887_WAVE12_PROOF = {"logical_index": 0,
 "payload_sha256": "eb9170128722472e627a9ae16fa23692c1fc19c3ed6156513bde1c98e0c45a91",
 "previous_request_id": "7825e8ef5b9181aeb184569692dadbe056e2dd63c9b33e00402529c4b5119e52",
 "previous_terminal_file_sha256": "8d40cc02f116d35e031a9d9f27a3d60875c59f58cd7877fd88155db544ac94dc",
 "previous_transport_progress_file_sha256": "0e2c2ee7932ec868f9813c5bdbbb79c30731166230abd66fecc0266cd1aace40",
 "relay_cooldown_clock": "not_before_unix",
 "relay_not_before_unix": 1795734464.15188,
 "newly_authorized_boundary_physical_attempts": 2,
 "execution_budget": {"first_logical_requests": 2,
                      "maximum_logical_requests": 4,
                      "maximum_physical_attempts": 8}}
RELAY_887_WAVE12_PREVIOUS_ATTEMPTS = ({"attempt": 0,
  "attempt_id": "675632bc2bc343dd81dc7f9db47ee122",
  "job_sha256": "1e8d448f693d58ef2188dbf43ecf03c71aa527efa1ab2176f2a6f66ee13346f7",
  "result_sha256": "b6f69ababedc0be666ff151e34fb06db260804510109bad3b3cf05df07df75b0",
  "error": "response_id_mismatch",
  "provider_dispatched": True,
  "ambiguous": True},
 {"attempt": 1,
  "attempt_id": "e9f779550dce4a6c88663a38d9e1dd0c",
  "job_sha256": "752160f18882c2b35e81529714690d7a8aceeaeac44bca186f2b6e76d7ec4aa2",
  "result_sha256": "7500a346b8f9713d5aea43bc1b892f5b98a41932cac94551959b3601260a88f8",
  "error": "response_id_mismatch",
  "provider_dispatched": True,
  "ambiguous": True})
RELAY_887_WAVE12_TERMINALS = ("d3a769a01f8a051033ec04170123c00d0f0d48ebe3ecae90206abee50ed8391b",
 "8d40cc02f116d35e031a9d9f27a3d60875c59f58cd7877fd88155db544ac94dc")
RELAY_887_WAVE12_REPLAY_FILES = {"attempt_00.replay_authorization.json": "23eaa9c35e511d785e75c542a6bd2aad14d89cb95e1b597decd7f6a01668e5a5",
 "attempt_00.replay.terminal.json": "6fcf8ccb4ef23554bab8dc153bfeb1f07dc1139826bbdf656db8416c1e1f47bf",
 "attempt_01.replay_link.json": "5c03d43f0cfe984293245b4730e9a8e27aab9f6866b5a630fd434e9133fbde10"}
RELAY_887_WAVE12_OBSERVATION = {"http_status": 200,
 "phase": "transport_failure",
 "error_type": "response_id_mismatch",
 "event_type": "response.output_item.added",
 "conflicting_event_type": "response.failed",
 "provider_error_code": "upstream_error",
 "sse_event_count": 41,
 "first_sse_seconds": 18.300744083011523,
 "last_sse_seconds": 195.47679645800963,
 "last_sse_activity_seconds": 195.47679645800963,
 "elapsed_seconds": 198.23884300002828,
 "model": "gpt-6.2-astra",
 "response_id": "resp_0d2e9864d776b2a0016b08ba3ccd0487d2bbcb54da9ac57ac0",
 "provider_request_id": "592b40d9-48c1-427d-9f7a-f9ca009cd83b",
 "conflicting_response_id": "resp_592b40d948c1427d9f7af9ca009cd83b",
 "response_id_matches_transport_alias": True,
 "sse_milestones": {"response.created": {"elapsed_seconds": 18.300744083011523, "interval_seconds": None},
                    "response.in_progress": {"elapsed_seconds": 18.30356779205613,
                                             "interval_seconds": 0.0028237090446054935},
                    "response.output_item.added": {"elapsed_seconds": 18.30534025002271,
                                                   "interval_seconds": 0.001772457966580987},
                    "response.output_item.done": {"elapsed_seconds": 27.846694374922663,
                                                  "interval_seconds": 9.541354124899954}}}
RELAY_887_WAVE12_FITTED_FILES = {"boundary.json": "5ab1672a1325e8cbe63a444d2846be756c8270a62450a80c25524b972a3ccb6c",
 "round_33/source.json": "51a79ba4929ff53020dfbcc5a22b596b1369ae638ac50cf051de5314ab2a2add",
 "round_33/boundary_memory.json": "a98e4a5b1f3fa54f5c1f2bbbf45056e1eec201d744478c8159f0f557c8b89fb0",
 "round_33/memory.json": "c8777ec320923c89d0a4a4481f2541ef2a879a43964b646f76e894c9a5632d05",
 "round_33/batch_freeze.json": "cbd42d6f573d569209bfa2c4b93f0266f29573fb154485d80790e745e4507a19",
 "round_33/selection.json": "1c8a22a407072bcfe69001505115db0f2e75f4e8ef620c372a131bfccb79dc61",
 "round_34/source.json": "6d2a8e1612104f6fc0299994f73cb69555fc0114bae88fcee84ebe3b06bf68ea",
 "round_34/transition.json": "bd0a36d764e2b7453b1df80e0d5a5ea081ae31e9853cee3769187a1e49d70b44",
 "round_34/fit_000/fit_audit.json": "644002146d84a4af6814bb8ac3631e84ceabf1a61bc8e2eedeb663a2e258c8a0",
 "round_34/fit_000/validation_metrics.json": "3b82ee8e65d6bbcfd862ef6bf3b10b39adfb445ff86abf583e3d227302ef1ee7",
 "round_34/authoritative_packet.json": "111a5842df2244af49b73ecc68bfd40e8394a490fc4d7cad4c6c8213f0d85b7f",
 "round_34/turn_00/request.json": "28547f479ce8fbe222852e6132eb169e06b31d3d72ad4e41c118a9787ce6d6cc"}
RELAY_887_WAVE12_SOURCES = {"studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/dispatcher.py": "cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/provider.py": "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/seed_runner.py": "417f41255d53d7794abcdc98c0a9258141d01c076df845bfa877013c84de1792",
 "studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py": "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py": "f6c5e6f58c92dd403e2fc08e5e2451a448a4507f087a30b3fd8cc580dcbcb251",
 "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_retry_successor.py": "6111c04c5c1eb6688e5cf6d0b086d4c1d8179a7bfcd7a9e9d04e8f572fce38de"}
RELAY_887_WAVE12_ARCHIVED_SSE_COUNTS = (41, 41)

RELAY_887_WAVE13_PREFIX_SHA = "fbe164551be532a60431aa8fabc3915a6088e1d3511e9d56f73c301906aaf945"
RELAY_887_WAVE13_FITTED_FILES = {"boundary.json": "69861ed5d2dd6b8962e7148e1a81877ac79665397d79ad1b4b6dd194db605ed5",
 "round_34/source.json": "6d2a8e1612104f6fc0299994f73cb69555fc0114bae88fcee84ebe3b06bf68ea",
 "round_34/boundary_memory.json": "c8777ec320923c89d0a4a4481f2541ef2a879a43964b646f76e894c9a5632d05",
 "round_34/boundary_user_message.json": "8ca8a6410f484ac4b0884a7dfd476741394a286d9f2e51612787c021e66219b2",
 "round_34/authoritative_packet.json": "111a5842df2244af49b73ecc68bfd40e8394a490fc4d7cad4c6c8213f0d85b7f",
 "round_34/turn_00/request.json": "5981946cd836d6ecb1a3aa08fb4503f09399c400b089844e6e89936e7316ef07"}

RELAY_887_WAVE14_PREFIX_SHA = "fbe24350696e3872fb9138a88d2ecf89e2c0f7b20639360d0ec74f3c068ace31"
RELAY_887_WAVE14_FITTED_FILES = {"boundary.json": "eae2b4f144373e39064a439c44188bbf7f93a0dc72ac72d92d68cf576e275e4a",
 "round_34/source.json": "6d2a8e1612104f6fc0299994f73cb69555fc0114bae88fcee84ebe3b06bf68ea",
 "round_34/boundary_memory.json": "c8777ec320923c89d0a4a4481f2541ef2a879a43964b646f76e894c9a5632d05",
 "round_34/boundary_user_message.json": "8ca8a6410f484ac4b0884a7dfd476741394a286d9f2e51612787c021e66219b2",
 "round_34/authoritative_packet.json": "111a5842df2244af49b73ecc68bfd40e8394a490fc4d7cad4c6c8213f0d85b7f",
 "round_34/turn_00/request.json": "59b8e44fd5681e65cb4fe9bb02d537125034349c69e0b3e3dab532a54dfab43d"}

RELAY_887_WAVE15_PREFIX_SHA = "c41feb781e892c9966df0da2dd6994f35139754e2e5f5b7d12b419a453d62a8a"
RELAY_887_WAVE15_FITTED_FILES = {"boundary.json": "6d5041836a0a1ebc90b5782de241fb1910f5efd5be6089876462f3527434f68f",
 "round_34/source.json": "6d2a8e1612104f6fc0299994f73cb69555fc0114bae88fcee84ebe3b06bf68ea",
 "round_34/boundary_memory.json": "c8777ec320923c89d0a4a4481f2541ef2a879a43964b646f76e894c9a5632d05",
 "round_34/boundary_user_message.json": "8ca8a6410f484ac4b0884a7dfd476741394a286d9f2e51612787c021e66219b2",
 "round_34/authoritative_packet.json": "111a5842df2244af49b73ecc68bfd40e8394a490fc4d7cad4c6c8213f0d85b7f",
 "round_34/turn_00/request.json": "2d1f61fcf068314ecb7124edf95dba659480cb552b700dc73577ee0f6667f7bd"}

REPAIR_6101_R8_RUN = f"{SERIAL_IPV4}-6101-r8-targeted-repair-01"
REPAIR_6101_R8_CONTRACT = "b618cd7fb5608e339b398ec89822e697e3565e09cb0b9c291afef330dd029c53"
REPAIR_6101_R8_AUTH = "cb982d841eb3c0eb4ad4268867de0e05e1e138bd875eec0487189432f7ddd485"
REPAIR_6101_R8_METADATA = "858d79ee9752b46b0803c19b0906d60b75ecf8664bbe45e5f436564bed1df892"
REPAIR_6101_R8_METADATA_FILE = "83bd8c1802e950e9e49cc5f100cf40bd03fe5a99bc4273e01911ae02c91313c6"
REPAIR_6101_R8_CHECKPOINT = "7f1cfcaf2a9275018d0a26b5819a34785acd8d2e502bf910f037085ee0d88da4"
REPAIR_6101_R8_FREEZE = "4b5a73dd172829fc5789fdf64eea96a577d7e9666c5d6a426a0f0ca26f5b3c14"
REPAIR_6101_R8_PARENT_FILES = {
    "seed_6101/round_08/turn_00/attempt_01.terminal.json": "014ea1b74f0a475e16492d7d4d7aa8e51cca986310cbbd19ef907d90cad03d8b",
    "seed_6101/round_08/turn_00/decision.json": "ee52b1d6f15c5cc6306909aabfd8fb74479417d675b5035aa2b7f81b10f8d882",
    "seed_6101/pause.request": "ae6a2801b839ae4e61cfc310cab353718ccc57ad538624351ac03121d23a167b",
    "workers/seed_6101.json": "c48b19f0271329016a11b940c5359c4b588445e09b197fed19f219299db79bf9",
}
REPAIR_6101_R8_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_6101_r8_targeted_repair.py":
        "ae0f3462d9cc22550b9368d0ca51940f16cd10f3e11f8e21b91036f2317a32b2",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_6101_r8_targeted_repair.py":
        "aee0c230bc11710cac958bf54c5c6b752e790ab017942e523e58fadcfde50dc3",
}
CONTINUATION_6101_R9_RUN = f"{SERIAL_IPV4}-6101-r9-continuation-01"
CONTINUATION_6101_R9_CONTRACT = "e63dff127877bed96403a8c4174dcddb1b43c68913ed60bae7631ce0b03be5c2"
CONTINUATION_6101_R9_AUTH = "989a6b41a0efc0363336bc9fa0af2b9c47f2d32eb52843a51b1f9341f7464768"
CONTINUATION_6101_R9_HANDOFF = "eba2171f2e112d1afaef89988bfcc7371556e428d17e4373864da72efa1bfa88"
CONTINUATION_6101_R9_REQUEST = "16c79ead1b244a0ff168db317e352815d3ce8694a7e326b298d8d5b8b57320d7"
CONTINUATION_6101_R9_SCOPE = {"maximum_logical_requests": 3, "maximum_physical_attempts": 2}
CONTINUATION_6101_R9_BUDGET = {"first_logical_requests": 27, "maximum_logical_requests": 81, "maximum_physical_attempts": 162}
CONTINUATION_6101_R9_DESCRIPTOR = {
    "round": 9,
    "memory_sha256": "734fc920ebbc1ea10010b64bccb7be9da3c5523e20378a6928555c58a1a903d5",
    "state_sha256": "8c283718a872f111d145e60296a769aa878c8c640ffae9017363fba503e266eb",
}
CONTINUATION_6101_R9_SOURCES = {
    **REPAIR_6101_R8_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_6101_r9_continuation.py":
        "7b94671f769237c9404a449956e1612d104ff78044dcfa3572b1ce101880a407",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_6101_r9_continuation.py":
        "f2b618b5bcc7e1d3de7870709b920c469bc30f8cc62efddda6b10ea9e9d68c2c",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/seed_runner.py":
        "193269dc7dab89e4e9c529338a130f224d331625b6646ed32e085ec97ca5005b",
}


CACHED_R10_6101_RUN = f"{SERIAL_IPV4}-6101-r10-cached-01"
CACHED_R10_6101_CONTRACT = "555ea880ed86c6e9347be1967874a9512c73aa8cd1142f90c343233828aa6f58"
CACHED_R10_6101_AUTH = "345ac4f06cb2940848743e053723780e6b31153002479b3607a936471864b45d"
CACHED_R10_6101_START = "dbeba66c3bd67937dfc0d5070104fb24a9c539960c8cb0f07c20efbec167b4fc"
CACHED_R10_6101_FIT = "07214f81948e1edbb1a7500cfa9e4dbe4604504a136defcd33f460f8fb842239"
CACHED_R10_6101_RECEIPT = "c996df723e45ee1161fa57c1704fbeda0c1e8d8f4aeb07da8c9506778bcc9c5f"
CACHED_R10_6101_BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
CACHED_R10_6101_FILES = {
    "cached_response_import.json": "52cdf6b6fc14d3f7ba4532351b7648b8a27627652b67f75f27a10b83b7b05117",
    "seed_6101/round_10/turn_00/attempt_00.terminal.json": "c942bc3bb66e934c7d8374729faef0c40ed6b83ed2d8246a8f49a213c34fd848",
    "seed_6101/round_10/batch_freeze.json": "7385ece32aeb66d8e1228451f23a154ab71aabb25e5c8900b53184513153ebcc",
    "seed_6101/round_10/selection.json": "0915aca51c0a7176d3daa8cb16a57cfe4e60c53d2e14b9278f27889a942b51c9",
    "seed_6101/round_10/memory.json": "ecad42f3fe8dab5f936bdf888d7cf7ca6b5db2f95e0105e127874b4f0862a109",
    "seed_6101/round_11/source.json": "13d6d959557a4ecffc79737cdd5b07dcabff22880472cb3d5b1c0ad49346ece5",
    "seed_6101/round_11/transition.json": "2490fc5d2537cc5ad92217d1f1d4cbedef5b70bbae03f951b231885738f95160",
    "seed_6101/round_11/authorized_feedback.json": "08ab76ad69e5aecd98cf2320ea5efc4f64fa064ed70bf394c080302ecc50a287",
    "seed_6101/round_11/fit_000/fit_audit.json": "f21f1774af7241b0098c1b5d40143847231c4dba10f6f4c63a13a22de11c0cf2",
    "seed_6101/round_11/fit_000/validation_metrics.json": "8594962b842aae8715ca4f883925070302089c03b8ecac0b1b7aea06964d6b3c",
}


CONTINUATION_6101_R11_RUN = f"{SERIAL_IPV4}-6101-r11-continuation-01"
CONTINUATION_6101_R11_CONTRACT = "bb0c4e008336863b71ec7f41be179c7e2b815c8bc6e23dfa317cb65eb046c553"
CONTINUATION_6101_R11_AUTH = "26acba8853492479b855727a687f8e80316494ee6173882b0328597fb3c957ed"
CONTINUATION_6101_R11_HANDOFF = "e520195193d963611590c717512412fce0eb9bc2916ab4c108dc400a6b6a4398"
CONTINUATION_6101_R11_REQUEST = "9538e3c0858c62e35d85121e70600acd14f30045d109276f3cf4e05f06ffd2cd"
CONTINUATION_6101_R11_BUDGET = {"first_logical_requests": 25, "maximum_logical_requests": 75, "maximum_physical_attempts": 150}
CONTINUATION_6101_R11_DESCRIPTOR = {
    "memory_sha256": "958041f961972e6a274f25712d484724116026fbbdd930acf72e6559d415b146",
    "round": 11,
    "state_sha256": "877489dcf354335b22f530309ca923c6ad38687d950e2ebab41d5a848124ac43",
}
CONTINUATION_6101_R11_SOURCES = {
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/seed_runner.py":
        "8368fc31377202dd4fb451a069cdfd28bcc88e08ca5ba44963cf187e18fc93e0",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/dispatch_v13_6101_r11_continuation.py":
        "7c648e1369badffc8dedbf00bc816741309e8ac1a7ed40ec77b3cf029fb9524c",
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_6101_r11_continuation.py":
        "f2a1f54826718a1f0f7cec6a6a691009debbd6fcf9048549f700190c74f45e44",
}


UNSTARTED_ANCHORS = {
    6101: "6959c9417166245403dad173ca82347336de53a37d643591ca449b0cf2a929c4",
    12203: "e056a9924b408e8885e7fa619f0ce63a44b8b3b47a2126d57c1fdc80e3df428d",
}

RELAY_887_WAVE13_SOURCES = {
    **RELAY_887_WAVE12_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "3b39e10c57829879752d06c6c06d3fbaa1baed68475a0a058376e3de810b3440",
}

RELAY_887_WAVE14_SOURCES = {
    **RELAY_887_WAVE13_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "41c3741137c15db593bd96992ee69c4cb2566c902f601165bb9d3edf56896761",
}

RELAY_887_WAVE15_SOURCES = {
    **RELAY_887_WAVE14_SOURCES,
    "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py":
        "911acf95aefff1b26e24d1e8612b33c0445253b22eae2d37ee282693c54530fb",
}

RELAY_887_WAVE13_CONTRACT = "aaaf2c4a13fcc3cc424e6640f66b951b61c467d2586ed5487b901bd2b954331c"
RELAY_887_WAVE13_AUTH = "f486dc5937a358afd903f075913c08a45c1743c7e51ae21920203f9b48c23145"
RELAY_887_WAVE13_PROOF = {
    **RELAY_887_WAVE12_PROOF,
    "previous_request_id": "2dbeab1c2e8bb4b6014752867f4a3820c72679d7d162339d846bc24eca2c2d28",
    "previous_terminal_file_sha256": "c696349b3e689cb6209f0babaa37e4b78dbb7744b1d7c44710634f09f15a197a",
    "previous_transport_progress_file_sha256": "8ced7e32c22d14974829f5d710bed2977290aa6236dae4bb5bcf7cfcc5579d37",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795735622.4724555,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 8},
}
RELAY_887_WAVE13_PARENT_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "c69c1ba0999b411f805fa72b57b8ce86",
        "job_sha256": "84459dfa1558654679414e19084b03378d150ba8c6bcd62761e0780a729e8f58",
        "result_sha256": "de08592429ce4b884d78291eac2c0d7447300f9e0aca5d9f99bdfba61d906c00",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "8367b4506c584f0ab07ce0fa907546c7",
        "job_sha256": "c56abfd88fce65c5d67e6d79ce1c0ee4474db10f0c3d23f908b7dbdccbd0fccb",
        "result_sha256": "f3796fd3113894ef09b602a338e496ae2d8423563be606dfeec7e765dc27a76c",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_887_WAVE13_PARENT_TERMINALS = (
    "1c8d55e2e7daef02633f12bdfb20583d104bef48adfef44d858b73ce3e5d73bd",
    "c696349b3e689cb6209f0babaa37e4b78dbb7744b1d7c44710634f09f15a197a",
)
RELAY_887_WAVE13_PARENT_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "8771790ee54488a4b8e27f5780b9b73f099b764db54f8e0afbb252eadaeaad3c",
    "attempt_00.replay_authorization.json": "6976a4dddaacad8a436b981f0536168969010191d3584654991fcb24b89a5c55",
    "attempt_01.replay_link.json": "082bec8aadfe1a33d1251594b9f67a4259e2cb2fda4c2991c93de551d9fe98ff",
}
RELAY_887_WAVE13_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "conflicting_event_type": "response.failed",
    "provider_error_code": "upstream_error",
    "sse_event_count": 41,
    "first_sse_seconds": 21.03567266697064,
    "last_sse_seconds": 198.78955541690812,
    "last_sse_activity_seconds": 198.78955541690812,
    "elapsed_seconds": 201.03768624993972,
    "model": "gpt-6.2-astra",
    "response_id": "resp_04d33df26e6482d3016b08bec71bc487d0acc2bab535a0231f",
    "provider_request_id": "468c1fd7-8ba5-4985-ac15-c9f3cb6290cb",
    "conflicting_response_id": "resp_468c1fd78ba54985ac15c9f3cb6290cb",
    "response_id_matches_transport_alias": True,
    "sse_milestones": {"response.created": {"elapsed_seconds": 21.03567266697064, "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 21.03834804194048,
                          "interval_seconds": 0.0026753749698400497},
 "response.output_item.added": {"elapsed_seconds": 21.047718417132273,
                                "interval_seconds": 0.009370375191792846},
 "response.output_item.done": {"elapsed_seconds": 30.417579208035022,
                               "interval_seconds": 9.369860790902749}},
}

RELAY_887_WAVE14_CONTRACT = "7eb07bbdbf1e8d310a295f649fa9fa270fe2f92aefae52c4f911221ae7311a22"
RELAY_887_WAVE14_AUTH = "02b08eb72ed69ab957fc2ec9968aaef9286fb20e91eec44d936a0cddc5d68212"
RELAY_887_WAVE14_PROOF = {
    **RELAY_887_WAVE13_PROOF,
    "previous_request_id": "c6f2997acf4fb0a661cb2994fff7a856336c563b0d6667e99e34d79ef3743a60",
    "previous_terminal_file_sha256": "e4cc3e8a9a24dc6436ad0ee3c318890759ec11681a615c86173fef65164223d1",
    "previous_transport_progress_file_sha256": "5708bc0b30cd909d9e7ecec426e85645e005148265c598f0a36f6b957a20aee4",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795736713.3270583,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 8},
}
RELAY_887_WAVE13_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "8df7bf378f1342f2952ca941e7f7d443",
        "job_sha256": "cc7bca7e149bba6ad6dfc05d8850b07139bc0590628f97b12157a2ffc10bf3e3",
        "result_sha256": "9c5c7fa65e56d5ebb9647114be5fe48bf0069842ed6fd4ed9e92949ea7e2f88d",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "6f0af1c51ba64c9dbd0dc74e16f93fb7",
        "job_sha256": "dae9ee02f83af5c634542072e1dfb50045daf551ef6642064f9e052c10dd980e",
        "result_sha256": "5f1bbbc577c0d96b3511e8bd7d43e7d7ccb7acb6f58fd729a97490ed63d877af",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_887_WAVE13_TERMINALS = (
    "57870f13bc097ba6e17ec953858b5b331833d20d6a00f4859c8bfb967e4e77cc",
    "e4cc3e8a9a24dc6436ad0ee3c318890759ec11681a615c86173fef65164223d1",
)
RELAY_887_WAVE13_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "5347dbd27ca1f1e073422c501ecf611f4fae6893b396e488ccacbf6df02ff4e1",
    "attempt_00.replay_authorization.json": "1b10bf47e27746810d9e3376e6a93b5fea9bdee5947d10af38e80970901a98ca",
    "attempt_01.replay_link.json": "2befb4fb4ab2885e9c2c5b755cbeb983c7102a85d51eeb6a8c8b156379553cf5",
}
RELAY_887_WAVE14_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "conflicting_event_type": "response.failed",
    "provider_error_code": "upstream_error",
    "sse_event_count": 41,
    "first_sse_seconds": 19.235291457967833,
    "last_sse_seconds": 196.5353928329423,
    "last_sse_activity_seconds": 196.5353928329423,
    "elapsed_seconds": 199.1754941670224,
    "model": "gpt-6.2-astra",
    "response_id": "resp_03f06fefadf5c43d016b08c3058b2087d1815e1daf9f0a329b",
    "provider_request_id": "683dd8bf-b82b-4309-ba87-6916cf08c937",
    "conflicting_response_id": "resp_683dd8bfb82b4309ba876916cf08c937",
    "response_id_matches_transport_alias": True,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.235291457967833,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.237216958077624,
                          "interval_seconds": 0.0019255001097917557},
 "response.output_item.added": {"elapsed_seconds": 19.23976770788431,
                                "interval_seconds": 0.002550749806687236},
 "response.output_item.done": {"elapsed_seconds": 28.552820875076577,
                               "interval_seconds": 9.313053167192265}},
}

RELAY_887_WAVE15_CONTRACT = "07115efa10ceaa7f3e3d58d9b036707e1e3ff859e2eb8705399d657582787c11"
RELAY_887_WAVE15_AUTH = "4515516c5ac03ddaed71db23ed36bc0efe607078f4e79eae313376ac1d4c6141"
RELAY_887_WAVE15_PROOF = {
    **RELAY_887_WAVE14_PROOF,
    "previous_request_id": "d2d7af468d0a51b3a1ecff95522f9c30ac55775ee70498c99ef72381c3cc0244",
    "previous_terminal_file_sha256": "df4e812be59f9b2ae132bfba4e5cb33c8344bd603e30c2b8890c9e4f614b51d7",
    "previous_transport_progress_file_sha256": "9ab3b6217f12b1944d29f043e8c2106dcc71068fbaec2d4a30eddab6992474e2",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795738639.839541,
    "newly_authorized_boundary_physical_attempts": 2,
    "execution_budget": {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 8},
}
RELAY_887_WAVE14_ATTEMPTS = (
    {
        "attempt": 0,
        "attempt_id": "de644c36140f438da43057111eec658d",
        "job_sha256": "183643c7b033baad6c8b9c32ebc662798ae56938e217cba43eff042e943c5274",
        "result_sha256": "17f9580354224eea46b1764864d6a6675a4cab80fa2fe6314a735ea09c55d778",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
    {
        "attempt": 1,
        "attempt_id": "86ed1130afb147fe818525c650de275e",
        "job_sha256": "f5b05cc3b23c64490a50d326956a723713a9ab78f3a803656b7a302a762b84e3",
        "result_sha256": "fe32ce0e28f6db3fa47d7642b13565fa779088652356972050e890afb8382246",
        "error": "response_id_mismatch",
        "provider_dispatched": True,
        "ambiguous": True,
    },
)
RELAY_887_WAVE14_TERMINALS = (
    "35cb0713357f835dc4cc0f1cb0cda33cdae0b29b29f4e141619d4b2f953482c5",
    "df4e812be59f9b2ae132bfba4e5cb33c8344bd603e30c2b8890c9e4f614b51d7",
)
RELAY_887_WAVE14_REPLAY_FILES = {
    "attempt_00.replay.terminal.json": "954f26c893133a4f48f80fe14544a8a613d34bc13f4ff611701b7c77d840e82d",
    "attempt_00.replay_authorization.json": "9645fc28e0baa82e0fe36a6a73d63ed5699efaf5d8ca0308d6ae838c4812f1c1",
    "attempt_01.replay_link.json": "e0f58753b39d50ece0e4ea51d16c88bad1761bcd513e2d54d5af180f24a537cc",
}
RELAY_887_WAVE15_OBSERVATION = {
    "http_status": 200,
    "phase": "transport_failure",
    "error_type": "response_id_mismatch",
    "event_type": "response.output_item.added",
    "sse_event_count": 41,
    "model": "gpt-6.2-astra",
    "provider_request_id": "6a2636ec-9934-4284-851f-a71277cac395",
    "response_id": "resp_021cc9ef0dcc7d5f016b08c6a724ec87d08690c3511de522e1",
    "conflicting_response_id": "resp_6a2636ec99344284851fa71277cac395",
    "conflicting_event_type": "response.failed",
    "response_id_matches_transport_alias": True,
    "provider_error_code": "upstream_error",
    "last_sse_seconds": 197.4295056250412,
    "last_sse_activity_seconds": 197.4295056250412,
    "elapsed_seconds": 199.8169130831957,
    "sse_milestones": {"response.created": {"elapsed_seconds": 19.828010083176196,
                      "interval_seconds": None},
 "response.in_progress": {"elapsed_seconds": 19.830502249998972,
                          "interval_seconds": 0.0024921668227761984},
 "response.output_item.added": {"elapsed_seconds": 19.83234287519008,
                                "interval_seconds": 0.0018406251911073923},
 "response.output_item.done": {"elapsed_seconds": 29.24828691710718,
                               "interval_seconds": 9.4159440419171}},
}
RELAY_887_WAVE15_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_887_WAVE14_ARCHIVED_SSE_COUNTS = (41, 41)
RELAY_887_WAVE13_ARCHIVED_SSE_COUNTS = (41, 41)
KEYS = ("key_01", "key_02", "key_03")
SHA = re.compile(r"[a-f0-9]{64}\Z")
IDENTITY = re.compile(r"[a-f0-9]{32}\Z")
FIT = re.compile(r"fit_[0-9]{3}\Z")
ROUND = re.compile(r"round_([0-9]{2})\Z")
CODE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.]{0,79}\Z")
PROVIDER_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
PROVIDER_DIAGNOSTICS = ("error_type", "conflicting_event_type", "provider_error_code", "provider_error_type",
                        "provider_error_param", "provider_incomplete_reason")
PHASES = {"request_dispatched", "http_phase", "http_headers", "sse_event",
          "transport_completed", "transport_failure"}
TOKEN_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def read_json(path):
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def checksum(path):
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def safe_code(value):
    return value if isinstance(value, str) and CODE.fullmatch(value) else None


def provider_token(value):
    if isinstance(value, str) and PROVIDER_ID.fullmatch(value) and not value.lower().startswith("sk-"):
        return value
    return None


def pid(value):
    return value if type(value) is int and value > 1 else None


def live(proc, *, coordinator=None, run=None):
    if proc is None:
        return False
    try:
        state = Path(f"/proc/{proc}/stat").read_text().rsplit(")", 1)[1].split()
        if state[0] == "Z" or coordinator is not None and int(state[1]) != coordinator:
            return False
        if run is None:
            return True
        argv = Path(f"/proc/{proc}/cmdline").read_bytes().split(b"\x00")
        return os.fsencode(run) in argv
    except (OSError, IndexError, ValueError):
        return False


def latest_round(directory):
    for path in sorted(directory.glob("round_[0-9][0-9]"), reverse=True):
        transition = read_json(path / "transition.json")
        fit = transition.get("fit_path")
        checkpoint = transition.get("checkpoint_sha256", "")
        if (not isinstance(fit, str) or not FIT.fullmatch(fit)
                or not isinstance(checkpoint, str) or not SHA.fullmatch(checkpoint)):
            continue
        evidence = {"source_sha256": path / "source.json",
                    "validation_metrics_sha256": path / fit / "validation_metrics.json"}
        if all(transition.get(field) and checksum(file) == transition[field] for field, file in evidence.items()):
            number = int(ROUND.fullmatch(path.name)[1])
            report = read_json(evidence["validation_metrics_sha256"])
            return number, {"checkpoint_sha256": checkpoint,
                            "validation_nrmse": report.get("combined_normalized_rmse")}
    if (directory / "initial/initial.json").is_file():
        return 0, {}
    return None, {}


def _relay_wait(turn, started, terminal, contract, delay):
    if (type(delay) is not int or delay != RELAY_RETRY_SECONDS or started.get("attempt") != 0
            or terminal.get("status") != "stopped" or terminal.get("ambiguous") is not True
            or (turn / "attempt_01.started.json").exists()
            or (turn / "attempt_00.recovery.terminal.json").exists()):
        return {}
    permit = read_json(turn / "attempt_00.replay_authorization.json")
    replay = read_json(turn / "attempt_00.replay.terminal.json")
    failed_unix = seconds(terminal.get("failed_unix"))
    prior_not_before = seconds(terminal.get("not_before_unix"))
    not_before = seconds(replay.get("not_before_unix"))
    terminal_sha = record_digest(terminal)
    if (failed_unix is None or prior_not_before is None or not_before is None
            or not_before != max(prior_not_before, failed_unix + delay)
            or replay.get("sealed_relay_retry_seconds") != delay
            or replay.get("status") != "retryable" or replay.get("ambiguous") is not False
            or replay.get("error") != "authorized_replay_unknown"
            or replay.get("supersedes_sha256") != terminal_sha
            or replay.get("replay_authorization_sha256") != record_digest(permit)
            or (replay.get("attempt_id"), replay.get("request_id"), replay.get("next_attempt"))
            != (started.get("attempt_id"), started.get("request_id"), 1)
            or (permit.get("attempt_id"), permit.get("request_id"), permit.get("next_attempt"))
            != (started.get("attempt_id"), started.get("request_id"), 1)
            or permit.get("contract_sha256") != contract
            or permit.get("decision") != "authorized_replay_unknown"
            or permit.get("original_terminal_sha256") != terminal_sha
            or permit.get("bridge_result_sha256") != terminal.get("bridge_result_sha256")
            or permit.get("accept_duplicate_billing_risk") is not True):
        return {}
    return {"relay_retry_wait_seconds": max(0, math.ceil(not_before - time.time()))}


def pending_attempt(run, directory, index, *, failed=False, relay_retry=None, contract=None):
    if index is None:
        return {}
    skipped = 0
    for path in sorted((directory / f"round_{index:02}").glob("turn_[0-9][0-9]/attempt_*.started.json"), reverse=True):
        started = read_json(path)
        identity = started.get("attempt_id", "")
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            continue
        stem = path.name.removesuffix(".started.json")
        recovery = path.with_name(f"{stem}.recovery.terminal.json")
        terminal = path.with_name(f"{stem}.terminal.json")
        record = read_json(recovery) or read_json(terminal)
        returned = (run / "bridge/results" / f"{identity}.json").is_file()
        if (failed and started.get("transport") == "mac_bridge" and not returned
                and not recovery.exists() and not terminal.exists() and not any(
                    (run / "bridge" / folder / f"{identity}.json").exists() for folder in ("outbox", "claims")
                )):
            skipped += 1
            continue
        status = safe_code(record.get("status")) or ("returned_from_mac" if returned else "waiting_for_mac")
        decision = read_json(path.parent / "decision.json")
        diagnosis = decision.get("diagnosis") or {}
        errors = Counter(code for entry in diagnosis.get("errors", ()) if isinstance(entry, dict)
                         if (code := safe_code(entry.get("code"))) is not None)
        return {"status": status, "failure": safe_code(record.get("error")), "ambiguous": record.get("ambiguous") is True,
                "round": index, "turn": int(path.parent.name[-2:]), "attempt": started.get("attempt"),
                "reserved_unix": seconds(started.get("started_unix")),
                "attempt_id": identity, "returned_from_mac": returned,
                **({"unsubmitted_reservations_skipped": skipped} if skipped else {}),
                **_relay_wait(path.parent, started, record, contract, relay_retry),
                "selection": safe_code(decision.get("state")), "selection_errors": dict(errors)}
    return {"status": "failed_before_bridge", "unsubmitted_reservations_skipped": skipped} if skipped else {}


def boundary_attempt_counts(run, directory, index, maximum):
    directory = directory / f"round_{index:02d}/turn_00"
    reserved = set()
    posted = set()
    for slot in range(maximum):
        started = read_json(directory / f"attempt_{slot:02d}.started.json")
        identity = started.get("attempt_id")
        if type(started.get("attempt")) is not int or started["attempt"] != slot or not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            continue
        reserved.add(identity)
        name = f"{identity}.json"
        job = read_json(run / "bridge/outbox" / name)
        result = read_json(run / "bridge/results" / name)
        if (job.get("attempt_id") == result.get("attempt_id") == identity
                and isinstance(sha := job.get("job_sha256"), str) and SHA.fullmatch(sha) and result.get("job_sha256") == sha
                and result.get("provider_dispatched") is True):
            posted.add(identity)
    return {"physical_attempts_reserved": len(reserved), "physical_attempts_posted_confirmed": len(posted),
            "physical_attempts_remaining": maximum - len(reserved)}


def record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def fallback_contract(run):
    record = read_json(run / "formal_contract.json")
    if record.get("run_directory") != str(run):
        return None
    actual = record_digest(record)
    approval = read_json(run / "authorization.json")
    return actual if approval.get("contract_sha256") == actual else None


def dispatch_contract(run, seed):
    binding = read_json(run / "operator/dispatch_binding.json")
    sha = binding.get("contract_sha256")
    if (binding.get("seed") == seed and binding.get("run") in (str(run), run.name)
            and isinstance(sha, str) and SHA.fullmatch(sha) and fallback_contract(run) == sha):
        return sha
    return None


def resilient_contract(runs, seed, previous, previous_contract, cohort_contract, follow_on_contract):
    run = runs / f"{SERIAL_IPV4}-{seed}-resilient-sse-01"
    record = read_json(run / "formal_contract.json")
    proof = record.get("predecessor") or {}
    profile = record.get("profile", {})
    parent = runs / (IPV4_CONTINUATION if seed == 2357 else SERIAL_IPV4)
    parent_contract = follow_on_contract if seed == 2357 else cohort_contract
    start = profile.get("resume_boundaries", {}).get(str(seed), {}).get("round")
    approval = read_json(run / "authorization.json")
    if (not previous_contract or not parent_contract
            or proof.get("dispatcher_module") != "dispatch_v13_resilient_sse"
            or proof.get("previous_run_directory") != str(previous)
            or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("parent_run_directory") != str(parent)
            or proof.get("parent_contract_sha256") != parent_contract
            or proof.get("seed") != seed or proof.get("round") != start or type(start) is not int
            or not 0 <= start < TARGET_ROUNDS
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != TARGET_LABELS
            or profile.get("ipv4_readiness_windows") != 3
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)):
        return None
    if contract := fallback_contract(run):
        return run, contract, start, profile["ipv4_readiness_windows"]
    return None


def resilient_failure_contract(runs, seed, previous, previous_contract, recovery_round):
    run = runs / f"{SERIAL_IPV4}-{seed}-resilient-failure-01"
    record = read_json(run / "formal_contract.json")
    proof = record.get("predecessor") or {}
    profile = record.get("profile", {})
    previous_record = read_json(previous / "formal_contract.json")
    previous_approval = read_json(previous / "authorization.json")
    approval = read_json(run / "authorization.json")
    if (previous.name != f"{SERIAL_IPV4}-{seed}-resilient-sse-01"
            or previous_approval.get("contract_sha256") != previous_contract
            or proof.get("dispatcher_module") != "dispatch_v13_resilient_failure_recovery"
            or proof.get("previous_run_directory") != str(previous)
            or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("previous_authorization_sha256") != record_digest(previous_approval)
            or proof.get("seed") != seed or proof.get("round") != recovery_round
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
            or profile != previous_record.get("profile")
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != TARGET_LABELS
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)):
        return None
    if contract := fallback_contract(run):
        return run, contract
    return None


def _signed_recovery_successor(run, seed, previous, previous_contract, dispatcher, physical_attempts, *,
                               previous_suffix="resilient-failure-01", target_labels=TARGET_LABELS):
    record = read_json(run / "formal_contract.json")
    proof = record.get("predecessor") or {}
    profile = record.get("profile", {})
    prior_record = read_json(previous / "formal_contract.json")
    prior_approval = read_json(previous / "authorization.json")
    approval = read_json(run / "authorization.json")
    boundary = profile.get("resume_boundaries", {}).get(str(seed), {})
    start = boundary.get("round")
    budget = proof.get("recovery_budget")
    if (previous.name != f"{SERIAL_IPV4}-{seed}-{previous_suffix}"
            or prior_approval.get("contract_sha256") != previous_contract
            or proof.get("dispatcher_module") != dispatcher
            or proof.get("previous_run_directory") != str(previous)
            or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("previous_authorization_sha256") != record_digest(prior_approval)
            or proof.get("seed") != seed or proof.get("round") != start or type(start) is not int
            or not 0 <= start < TARGET_ROUNDS
            or budget != {"maximum_logical_requests": 1, "maximum_physical_attempts": physical_attempts}
            or any(type(value) is not int for value in budget.values())
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != target_labels
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)):
        return None
    prior_round, _ = latest_round(previous / f"seed_{seed}")
    if prior_round is None:
        prior_round = prior_record.get("profile", {}).get("resume_boundaries", {}).get(str(seed), {}).get("round")
    if prior_round != start:
        return None
    if contract := fallback_contract(run):
        return contract, start, profile, proof
    return None


def relay_retry_contract(runs, seed, previous, previous_contract):
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-01"
    if candidate := _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_relay_retry", 2):
        contract, start, profile, _ = candidate
        if (type(profile.get("sealed_relay_retry_seconds")) is int
                and profile["sealed_relay_retry_seconds"] == RELAY_RETRY_SECONDS
                and profile.get("audited_future_transport_replay") is True
                and profile.get("replay_eligible_from_round") == start
                and profile.get("ipv4_readiness_windows") == 3):
            return run, contract, start, profile["ipv4_readiness_windows"]
    return None


def relay_retry_02_contract(runs, seed, previous, previous_contract):
    if seed == 6101:
        return relay_retry_02_6101_contract(runs, previous, previous_contract)
    if seed == 2357:
        return relay_retry_02_2357_contract(runs, previous, previous_contract)
    if seed == 157:
        signed_contract, signed_approval = RELAY_157_WAVE01_CONTRACT, RELAY_157_WAVE01_AUTH
        deadline, boundary_round = RELAY_157_NOT_BEFORE, 3
        boundary_proof = {
            "prior_boundary_budget": {"physical_authorized": 2, "physical_already_used": 2, "physical_remaining": 0},
            "newly_authorized_boundary_physical_attempts": 1,
            "previous_request_id": RELAY_157_BOUNDARY_REQUEST,
            "execution_budget": {"first_logical_requests": 33, "maximum_logical_requests": 97,
                                 "maximum_physical_attempts": 193},
        }
    elif seed == 12203:
        signed_contract, signed_approval = RELAY_12203_WAVE01_CONTRACT, RELAY_12203_WAVE01_AUTH
        deadline, boundary_round = RELAY_12203_NOT_BEFORE, 4
        boundary_proof = {
            "inherited_boundary_budget": {"physical_authorized": 2, "physical_already_used": 1, "physical_remaining": 1},
        }
    else:
        return None
    if previous_contract != signed_contract:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-02"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract,
                                           f"dispatch_v13_{seed}_relay_retry", 1, previous_suffix="relay-retry-01")
    if candidate is None:
        return None
    contract, start, profile, proof = candidate
    not_before = seconds(proof.get("relay_not_before_unix"))
    if (start == boundary_round and proof.get("previous_authorization_sha256") == signed_approval
            and record_digest({name: proof.get(name) for name in boundary_proof}) == record_digest(boundary_proof)
            and not_before == deadline and profile.get("audited_future_transport_replay") is True
            and profile.get("replay_eligible_from_round") == start + 1
            and type(profile.get("sealed_relay_retry_seconds")) is int
            and profile["sealed_relay_retry_seconds"] == RELAY_RETRY_SECONDS
            and profile.get("ipv4_readiness_windows") == 3):
        return run, contract, start, profile["ipv4_readiness_windows"], not_before
    return None


def relay_retry_02_6101_contract(runs, previous, previous_contract):
    seed = 6101
    if previous_contract != RELAY_6101_WAVE01_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-02"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="relay-retry-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old_profile = read_json(previous / "formal_contract.json").get("profile", {})
    boundaries = profile.get("resume_boundaries", {})
    exact = {**RELAY_6101_WAVE02_PROOF, "previous_authorization_sha256": RELAY_6101_WAVE01_AUTH,
             "reviewed_fitted_prefix_files": RELAY_6101_FITTED_PREFIX}
    if (start != 2 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 1
            or set(boundaries) != {str(seed)} or profile != {**old_profile, "resume_boundaries": boundaries,
                                                              "replay_eligible_from_round": 3}
            or profile.get("audited_future_transport_replay") is not True
            or type(profile.get("sealed_relay_retry_seconds")) is not int
            or profile["sealed_relay_retry_seconds"] != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    directory = previous / f"seed_{seed}"
    receipts = _reviewed_fitted_relay_receipts(directory, start, proof, RELAY_6101_FITTED_PREFIX,
                                              RELAY_6101_WAVE01_ATTEMPTS, RELAY_6101_WAVE01_TERMINALS)
    if receipts is None:
        return None
    chain = [_relay_review(previous, previous_contract, RELAY_6101_WAVE01_AUTH, proof, receipts)]
    if proof.get("reviewed_retry_chain") != chain:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def relay_retry_02_2357_contract(runs, previous, previous_contract):
    seed = 2357
    if previous_contract != RELAY_2357_WAVE01_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-02"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="relay-retry-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old_profile = read_json(previous / "formal_contract.json").get("profile", {})
    sources = read_json(run / "formal_contract.json").get("source_sha256") or {}
    boundaries = profile.get("resume_boundaries", {})
    exact = {**RELAY_2357_WAVE02_PROOF, "previous_authorization_sha256": RELAY_2357_WAVE01_AUTH,
             "reviewed_fitted_prefix_files": RELAY_2357_FITTED_PREFIX,
             "previous_transport_observation": RELAY_2357_WAVE02_OBSERVATION}
    if (start != 4 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 4
            or set(boundaries) != {str(seed)} or profile != {**old_profile, "resume_boundaries": boundaries,
                                                              "replay_eligible_from_round": 5}
            or profile.get("audited_future_transport_replay") is not True
            or type(profile.get("sealed_relay_retry_seconds")) is not int
            or profile["sealed_relay_retry_seconds"] != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or any(sources.get(name) != sha for name, sha in RELAY_RETRY_QUEUE_SOURCES.items())
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    directory = previous / f"seed_{seed}"
    turn = directory / f"round_{start:02d}/turn_00"
    if any(checksum(turn / name) != sha for name, sha in RELAY_2357_WAVE01_REPLAY_FILES.items()):
        return None
    receipts = _reviewed_fitted_relay_receipts(directory, start, proof, RELAY_2357_FITTED_PREFIX,
                                              RELAY_2357_WAVE01_ATTEMPTS, RELAY_2357_WAVE01_TERMINALS)
    if receipts is None or proof.get("reviewed_retry_chain") != [_relay_review(previous, previous_contract,
                                                                                RELAY_2357_WAVE01_AUTH, proof, receipts)]:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def _reviewed_fitted_relay_receipts(directory, start, proof, fitted_prefix, expected_attempts, terminal_shas):
    if any(checksum(directory / name) != sha for name, sha in fitted_prefix.items()):
        return None
    receipts = proof.get("previous_attempts")
    if type(receipts) is not list or len(receipts) != len(expected_attempts):
        return None
    turn = directory / f"round_{start:02d}/turn_00"
    for receipt, expected, sha in zip(receipts, expected_attempts, terminal_shas, strict=True):
        terminal = turn / f"attempt_{expected['attempt']:02d}.terminal.json"
        if (type(receipt) is not dict or checksum(terminal) != sha
                or receipt.get("terminal_sha256") != record_digest(read_json(terminal))
                or record_digest({key: receipt.get(key) for key in expected}) != record_digest(expected)):
            return None
    return receipts


def _relay_review(previous, contract, approval, proof, attempts):
    return {"contract_sha256": contract, "authorization_sha256": approval, "run_directory": str(previous),
            "request_id": proof["previous_request_id"], "terminal_file_sha256": proof["previous_terminal_file_sha256"],
            "progress_file_sha256": proof["previous_transport_progress_file_sha256"], "previous_attempts": attempts}


def _reviewed_relay_prefix(runs, seed, wave, old):
    if (seed, wave) in ((157, 9), (2357, 8), (6101, 9), (6101, 10), (887, 10), (887, 11), (887, 12)):
        # A new fitted round starts a new replay ledger. The signed predecessor retains earlier rounds.
        return []
    if seed == 887 and 13 <= wave <= 15:
        prefix = old.get("reviewed_retry_chain")
        expected = (
            (CONTINUATION_887_R33_RUN, CONTINUATION_887_R33_CONTRACT, CONTINUATION_887_R33_AUTH, RELAY_887_WAVE12_PROOF),
            (f"{SERIAL_IPV4}-887-relay-retry-12", RELAY_887_WAVE12_CONTRACT, RELAY_887_WAVE12_AUTH, RELAY_887_WAVE13_PROOF),
            (f"{SERIAL_IPV4}-887-relay-retry-13", RELAY_887_WAVE13_CONTRACT, RELAY_887_WAVE13_AUTH, RELAY_887_WAVE14_PROOF),
        )[:wave - 12]
        if type(prefix) is not list or len(prefix) != len(expected):
            return None
        for review, (name, contract, approval, proof) in zip(prefix, expected, strict=True):
            attempts = review.get("previous_attempts") if isinstance(review, dict) else None
            if (type(attempts) is not list or len(attempts) != 2
                    or review != _relay_review(runs / name, contract, approval, proof, attempts)):
                return None
        return prefix
    if seed == 887 and wave in (6, 7, 8, 9):
        prefix = old.get("reviewed_retry_chain")
        expected = [(CONTINUATION_887_R9_CONTRACT, CONTINUATION_887_R9_AUTH, CONTINUATION_887_R9_RUN)]
        if wave >= 7:
            expected.append((RELAY_887_WAVE05_CONTRACT, RELAY_887_WAVE05_AUTH, f"{SERIAL_IPV4}-887-relay-retry-05"))
        if wave >= 8:
            expected.append((RELAY_887_WAVE06_CONTRACT, RELAY_887_WAVE06_AUTH, f"{SERIAL_IPV4}-887-relay-retry-06"))
        if wave == 9:
            expected.append((RELAY_887_WAVE07_CONTRACT, RELAY_887_WAVE07_AUTH, f"{SERIAL_IPV4}-887-relay-retry-07"))
        if type(prefix) is not list or len(prefix) != len(expected):
            return None
        for prior, (contract, auth, name) in zip(prefix, expected, strict=True):
            if (type(prior) is not dict
                    or (prior.get("contract_sha256"), prior.get("authorization_sha256"), prior.get("run_directory"))
                    != (contract, auth, str(runs / name))
                    or type(prior.get("previous_attempts")) is not list or len(prior["previous_attempts"]) != 1):
                return None
        return prefix
    if wave >= 4 or seed in (6101, 887, 2357):
        prefix = old.get("reviewed_retry_chain")
        first_wave = 1
        if (seed, wave) == (157, 8):
            first_wave = 6
            expected = ((CACHED_R3_157_CONTRACT, CACHED_R3_157_AUTH, 1),)
        elif (seed, wave) == (6101, 8):
            first_wave = 6
            expected = ((CACHED_R2_6101_CONTRACT, CACHED_R2_6101_AUTH, 1),)
        elif seed == 2357 and wave in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
            first_wave = 7
            expected = ((RELAY_2357_WAVE07_CONTRACT, RELAY_2357_WAVE07_AUTH, 2),
                        (RELAY_2357_WAVE08_CONTRACT, RELAY_2357_WAVE08_AUTH, 2),
                        (RELAY_2357_WAVE09_CONTRACT, RELAY_2357_WAVE09_AUTH, 1),
                        (RELAY_2357_WAVE10_CONTRACT, RELAY_2357_WAVE10_AUTH, 1),
                        (RELAY_2357_WAVE11_CONTRACT, RELAY_2357_WAVE11_AUTH, 2),
                        (RELAY_2357_WAVE12_CONTRACT, RELAY_2357_WAVE12_AUTH, 2),
                        (RELAY_2357_WAVE13_CONTRACT, RELAY_2357_WAVE13_AUTH, 2),
                        (RELAY_2357_WAVE14_CONTRACT, RELAY_2357_WAVE14_AUTH, 2),
                        (RELAY_2357_WAVE15_CONTRACT, RELAY_2357_WAVE15_AUTH, 2),
                        (RELAY_2357_WAVE16_CONTRACT, RELAY_2357_WAVE16_AUTH, 2),
                        (RELAY_2357_WAVE17_CONTRACT, RELAY_2357_WAVE17_AUTH, 2),
                        (RELAY_2357_WAVE18_CONTRACT, RELAY_2357_WAVE18_AUTH, 2),
                        (RELAY_2357_WAVE19_CONTRACT, RELAY_2357_WAVE19_AUTH, 2),
                        (RELAY_2357_WAVE20_CONTRACT, RELAY_2357_WAVE20_AUTH, 2),
                        (RELAY_2357_WAVE21_CONTRACT, RELAY_2357_WAVE21_AUTH, 2),
                        (RELAY_2357_WAVE22_CONTRACT, RELAY_2357_WAVE22_AUTH, 2),
                        (RELAY_2357_WAVE23_CONTRACT, RELAY_2357_WAVE23_AUTH, 2),
                        (RELAY_2357_WAVE24_CONTRACT, RELAY_2357_WAVE24_AUTH, 2),
                        (RELAY_2357_WAVE25_CONTRACT, RELAY_2357_WAVE25_AUTH, 2),
                        (RELAY_2357_WAVE26_CONTRACT, RELAY_2357_WAVE26_AUTH, 2),
                        (RELAY_2357_WAVE27_CONTRACT, RELAY_2357_WAVE27_AUTH, 2),
                        (RELAY_2357_WAVE28_CONTRACT, RELAY_2357_WAVE28_AUTH, 2),
                        (RELAY_2357_WAVE29_CONTRACT, RELAY_2357_WAVE29_AUTH, 2))
        elif seed == 6101 and wave in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28):
            first_wave = 9
            expected = ((RELAY_6101_WAVE09_CONTRACT, RELAY_6101_WAVE09_AUTH, 2),
                        (RELAY_6101_WAVE10_CONTRACT, RELAY_6101_WAVE10_AUTH, 2),
                        (RELAY_6101_WAVE11_CONTRACT, RELAY_6101_WAVE11_AUTH, 1),
                        (RELAY_6101_WAVE12_CONTRACT, RELAY_6101_WAVE12_AUTH, 1),
                        (RELAY_6101_WAVE13_CONTRACT, RELAY_6101_WAVE13_AUTH, 2),
                        (RELAY_6101_WAVE14_CONTRACT, RELAY_6101_WAVE14_AUTH, 2),
                        (RELAY_6101_WAVE15_CONTRACT, RELAY_6101_WAVE15_AUTH, 2),
                        (RELAY_6101_WAVE16_CONTRACT, RELAY_6101_WAVE16_AUTH, 2),
                        (RELAY_6101_WAVE17_CONTRACT, RELAY_6101_WAVE17_AUTH, 1),
                        (RELAY_6101_WAVE18_CONTRACT, RELAY_6101_WAVE18_AUTH, 1),
                        (RELAY_6101_WAVE19_CONTRACT, RELAY_6101_WAVE19_AUTH, 2),
                        (RELAY_6101_WAVE20_CONTRACT, RELAY_6101_WAVE20_AUTH, 2),
                        (RELAY_6101_WAVE21_CONTRACT, RELAY_6101_WAVE21_AUTH, 2),
                        (RELAY_6101_WAVE22_CONTRACT, RELAY_6101_WAVE22_AUTH, 2),
                        (RELAY_6101_WAVE23_CONTRACT, RELAY_6101_WAVE23_AUTH, 2),
                        (RELAY_6101_WAVE24_CONTRACT, RELAY_6101_WAVE24_AUTH, 2),
                        (RELAY_6101_WAVE25_CONTRACT, RELAY_6101_WAVE25_AUTH, 2),
                        (RELAY_6101_WAVE26_CONTRACT, RELAY_6101_WAVE26_AUTH, 2))
        elif seed == 157 and wave in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
            first_wave = 8
            expected = ((RELAY_157_WAVE08_CONTRACT, RELAY_157_WAVE08_AUTH, 1),
                        (RELAY_157_WAVE09_CONTRACT, RELAY_157_WAVE09_AUTH, 2),
                        (RELAY_157_WAVE10_CONTRACT, RELAY_157_WAVE10_AUTH, 1),
                        (RELAY_157_WAVE11_CONTRACT, RELAY_157_WAVE11_AUTH, 1),
                        (RELAY_157_WAVE12_CONTRACT, RELAY_157_WAVE12_AUTH, 2),
                        (RELAY_157_WAVE13_CONTRACT, RELAY_157_WAVE13_AUTH, 2),
                        (RELAY_157_WAVE14_CONTRACT, RELAY_157_WAVE14_AUTH, 2),
                        (RELAY_157_WAVE15_CONTRACT, RELAY_157_WAVE15_AUTH, 2),
                        (RELAY_157_WAVE16_CONTRACT, RELAY_157_WAVE16_AUTH, 2),
                        (RELAY_157_WAVE17_CONTRACT, RELAY_157_WAVE17_AUTH, 2),
                        (RELAY_157_WAVE18_CONTRACT, RELAY_157_WAVE18_AUTH, 2),
                        (RELAY_157_WAVE19_CONTRACT, RELAY_157_WAVE19_AUTH, 2),
                        (RELAY_157_WAVE20_CONTRACT, RELAY_157_WAVE20_AUTH, 2),
                        (RELAY_157_WAVE21_CONTRACT, RELAY_157_WAVE21_AUTH, 2),
                        (RELAY_157_WAVE22_CONTRACT, RELAY_157_WAVE22_AUTH, 2),
                        (RELAY_157_WAVE23_CONTRACT, RELAY_157_WAVE23_AUTH, 2),
                        (RELAY_157_WAVE24_CONTRACT, RELAY_157_WAVE24_AUTH, 2),
                        (RELAY_157_WAVE25_CONTRACT, RELAY_157_WAVE25_AUTH, 2),
                        (RELAY_157_WAVE26_CONTRACT, RELAY_157_WAVE26_AUTH, 2),
                        (RELAY_157_WAVE27_CONTRACT, RELAY_157_WAVE27_AUTH, 2),
                        (RELAY_157_WAVE28_CONTRACT, RELAY_157_WAVE28_AUTH, 2),
                        (RELAY_157_WAVE29_CONTRACT, RELAY_157_WAVE29_AUTH, 2))
        elif seed == 12203 and 8 <= wave <= 37:
            first_wave = 6
            expected = ((RELAY_12203_WAVE06_CONTRACT, RELAY_12203_WAVE06_AUTH, 2),
                        (RELAY_12203_WAVE07_CONTRACT, RELAY_12203_WAVE07_AUTH, 2),
                        (RELAY_12203_WAVE08_CONTRACT, RELAY_12203_WAVE08_AUTH, 2),
                        (RELAY_12203_WAVE09_CONTRACT, RELAY_12203_WAVE09_AUTH, 1),
                        (RELAY_12203_WAVE10_CONTRACT, RELAY_12203_WAVE10_AUTH, 1),
                        (RELAY_12203_WAVE11_CONTRACT, RELAY_12203_WAVE11_AUTH, 2),
                        (RELAY_12203_WAVE12_CONTRACT, RELAY_12203_WAVE12_AUTH, 2),
                        (RELAY_12203_WAVE13_CONTRACT, RELAY_12203_WAVE13_AUTH, 1),
                        (RELAY_12203_WAVE14_CONTRACT, RELAY_12203_WAVE14_AUTH, 1),
                        (RELAY_12203_WAVE15_CONTRACT, RELAY_12203_WAVE15_AUTH, 1),
                        (RELAY_12203_WAVE16_CONTRACT, RELAY_12203_WAVE16_AUTH, 1),
                        (RELAY_12203_WAVE17_CONTRACT, RELAY_12203_WAVE17_AUTH, 1),
                        (RELAY_12203_WAVE18_CONTRACT, RELAY_12203_WAVE18_AUTH, 2),
                        (RELAY_12203_WAVE19_CONTRACT, RELAY_12203_WAVE19_AUTH, 2),
                        (RELAY_12203_WAVE20_CONTRACT, RELAY_12203_WAVE20_AUTH, 2),
                        (RELAY_12203_WAVE21_CONTRACT, RELAY_12203_WAVE21_AUTH, 2),
                        (RELAY_12203_WAVE22_CONTRACT, RELAY_12203_WAVE22_AUTH, 2),
                        (RELAY_12203_WAVE23_CONTRACT, RELAY_12203_WAVE23_AUTH, 2),
                        (RELAY_12203_WAVE24_CONTRACT, RELAY_12203_WAVE24_AUTH, 2),
                        (RELAY_12203_WAVE25_CONTRACT, RELAY_12203_WAVE25_AUTH, 2),
                        (RELAY_12203_WAVE26_CONTRACT, RELAY_12203_WAVE26_AUTH, 2),
                        (RELAY_12203_WAVE27_CONTRACT, RELAY_12203_WAVE27_AUTH, 2),
                        (RELAY_12203_WAVE28_CONTRACT, RELAY_12203_WAVE28_AUTH, 2),
                        (RELAY_12203_WAVE29_CONTRACT, RELAY_12203_WAVE29_AUTH, 2),
                        (RELAY_12203_WAVE30_CONTRACT, RELAY_12203_WAVE30_AUTH, 2),
                        (RELAY_12203_WAVE31_CONTRACT, RELAY_12203_WAVE31_AUTH, 2),
                        (RELAY_12203_WAVE32_CONTRACT, RELAY_12203_WAVE32_AUTH, 2),
                        (RELAY_12203_WAVE33_CONTRACT, RELAY_12203_WAVE33_AUTH, 2),
                        (RELAY_12203_WAVE34_CONTRACT, RELAY_12203_WAVE34_AUTH, 2),
                        (RELAY_12203_WAVE35_CONTRACT, RELAY_12203_WAVE35_AUTH, 2))
        elif seed == 157:
            expected = ((RELAY_157_WAVE01_CONTRACT, RELAY_157_WAVE01_AUTH, 2),
                        (RELAY_157_WAVE02_CONTRACT, RELAY_157_WAVE02_AUTH, 1),
                        (RELAY_157_WAVE03_CONTRACT, RELAY_157_WAVE03_AUTH, 1),
                        (RELAY_157_WAVE04_CONTRACT, RELAY_157_WAVE04_AUTH, 1))
        elif seed == 12203:
            expected = ((RELAY_12203_WAVE01_CONTRACT, RELAY_12203_WAVE01_AUTH, 1),
                        (RELAY_12203_WAVE02_CONTRACT, RELAY_12203_WAVE02_AUTH, 1),
                        (RELAY_12203_WAVE03_CONTRACT, RELAY_12203_WAVE03_AUTH, 1),
                        (RELAY_12203_WAVE04_CONTRACT, RELAY_12203_WAVE04_AUTH, 1))
        elif seed == 887:
            expected = ((RELAY_887_CACHED_CONTRACT, RELAY_887_CACHED_AUTH, 1),
                        (RELAY_887_WAVE02_CONTRACT, RELAY_887_WAVE02_AUTH, 1))
        elif seed == 2357:
            expected = ((RELAY_2357_WAVE01_CONTRACT, RELAY_2357_WAVE01_AUTH, 2),
                        (RELAY_2357_WAVE02_CONTRACT, RELAY_2357_WAVE02_AUTH, 1),
                        (RELAY_2357_WAVE03_CONTRACT, RELAY_2357_WAVE03_AUTH, 2),
                        (RELAY_2357_WAVE04_CONTRACT, RELAY_2357_WAVE04_AUTH, 1),
                        (RELAY_2357_WAVE05_CONTRACT, RELAY_2357_WAVE05_AUTH, 1))
        else:
            expected = ((RELAY_6101_WAVE01_CONTRACT, RELAY_6101_WAVE01_AUTH, 2),
                        (RELAY_6101_WAVE02_CONTRACT, RELAY_6101_WAVE02_AUTH, 1),
                        (RELAY_6101_WAVE03_CONTRACT, RELAY_6101_WAVE03_AUTH, 1),
                        (RELAY_6101_WAVE04_CONTRACT, RELAY_6101_WAVE04_AUTH, 1))
        expected = expected[:wave - first_wave - 1]
        if type(prefix) is not list or len(prefix) != len(expected):
            return None
        for index, (review, (contract, approval, count)) in enumerate(zip(prefix, expected, strict=True), first_wave):
            if (seed, wave) == (157, 8):
                suffix = "cached-r3-01"
            elif (seed, wave) == (6101, 8):
                suffix = "cached-r2-01"
            elif seed == 887 and index == 1:
                suffix = "cached-r7-01"
            else:
                suffix = f"relay-retry-{index:02d}"
            reviewed_run = runs / f"{SERIAL_IPV4}-{seed}-{suffix}"
            if (type(review) is not dict or review.get("run_directory") != str(reviewed_run)
                    or (review.get("contract_sha256"), review.get("authorization_sha256")) != (contract, approval)
                    or type(review.get("previous_attempts")) is not list or len(review["previous_attempts"]) != count):
                return None
        return prefix
    if seed == 157:
        wave01_contract, wave01_auth, wave01_count = RELAY_157_WAVE01_CONTRACT, RELAY_157_WAVE01_AUTH, 2
    else:
        wave01_contract, wave01_auth, wave01_count = RELAY_12203_WAVE01_CONTRACT, RELAY_12203_WAVE01_AUTH, 1
    old_attempts = old.get("previous_attempts")
    terminals = old.get("previous_terminal_file_sha256")
    traces = old.get("previous_transport_progress_file_sha256")
    if wave01_count == 1:
        terminals, traces = [terminals], [traces]
    if (type(old_attempts) is not list or len(old_attempts) != wave01_count
            or any(type(item) is not dict or type(item.get("attempt")) is not int or item["attempt"] != index
                   for index, item in enumerate(old_attempts))
            or any(type(hashes) is not list or len(hashes) != wave01_count
                   or any(not isinstance(value, str) or not SHA.fullmatch(value) for value in hashes)
                   for hashes in (terminals, traces))):
        return None
    request_id = old.get("previous_request_id")
    if not isinstance(request_id, str) or not SHA.fullmatch(request_id):
        return None
    return [{"contract_sha256": wave01_contract, "authorization_sha256": wave01_auth,
             "run_directory": str(runs / f"{SERIAL_IPV4}-{seed}-relay-retry-01"), "request_id": request_id,
             "terminal_file_sha256": terminals[-1], "progress_file_sha256": traces[-1], "previous_attempts": old_attempts}]


def reviewed_relay_contract(runs, seed, wave, previous, previous_contract):
    if (seed, wave) == (12203, 7):
        return relay_retry_07_12203_contract(runs, previous, previous_contract)
    if (seed, wave) == (157, 3):
        pinned_contract, pinned_auth = RELAY_157_WAVE02_CONTRACT, RELAY_157_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 3, RELAY_157_WAVE02_ATTEMPT, RELAY_157_WAVE03_PROOF
    elif (seed, wave) == (12203, 3):
        pinned_contract, pinned_auth = RELAY_12203_WAVE02_CONTRACT, RELAY_12203_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_12203_WAVE02_ATTEMPT, RELAY_12203_WAVE03_PROOF
    elif (seed, wave) == (6101, 3):
        pinned_contract, pinned_auth = RELAY_6101_WAVE02_CONTRACT, RELAY_6101_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 2, RELAY_6101_WAVE02_ATTEMPT, RELAY_6101_WAVE03_PROOF
    elif (seed, wave) == (6101, 4):
        pinned_contract, pinned_auth = RELAY_6101_WAVE03_CONTRACT, RELAY_6101_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 2, RELAY_6101_WAVE03_ATTEMPT, RELAY_6101_WAVE04_PROOF
    elif (seed, wave) == (6101, 5):
        pinned_contract, pinned_auth = RELAY_6101_WAVE04_CONTRACT, RELAY_6101_WAVE04_AUTH
        boundary, prior_attempt, wave_proof = 2, RELAY_6101_WAVE04_ATTEMPT, RELAY_6101_WAVE05_PROOF
    elif (seed, wave) == (6101, 6):
        pinned_contract, pinned_auth = RELAY_6101_WAVE05_CONTRACT, RELAY_6101_WAVE05_AUTH
        boundary, prior_attempt, wave_proof = 2, RELAY_6101_WAVE05_ATTEMPT, RELAY_6101_WAVE06_PROOF
    elif (seed, wave) == (6101, 8):
        pinned_contract, pinned_auth = RELAY_6101_WAVE07_CONTRACT, RELAY_6101_WAVE07_AUTH
        boundary, prior_attempt, wave_proof = 3, RELAY_6101_WAVE07_ATTEMPT, RELAY_6101_WAVE08_PROOF
    elif (seed, wave) == (6101, 9):
        pinned_contract, pinned_auth = CONTINUATION_6101_R5_CONTRACT, CONTINUATION_6101_R5_AUTH
        boundary, prior_attempt, wave_proof = 6, RELAY_6101_WAVE09_PRIOR_ATTEMPT, RELAY_6101_WAVE09_PROOF
    elif (seed, wave) == (6101, 10):
        pinned_contract, pinned_auth = RELAY_6101_WAVE09_CONTRACT, RELAY_6101_WAVE09_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE10_PROOF
    elif (seed, wave) == (6101, 11):
        pinned_contract, pinned_auth = RELAY_6101_WAVE10_CONTRACT, RELAY_6101_WAVE10_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE11_PROOF
    elif (seed, wave) == (6101, 12):
        pinned_contract, pinned_auth = RELAY_6101_WAVE11_CONTRACT, RELAY_6101_WAVE11_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_6101_WAVE11_ATTEMPT, RELAY_6101_WAVE12_PROOF
    elif (seed, wave) == (6101, 13):
        pinned_contract, pinned_auth = RELAY_6101_WAVE12_CONTRACT, RELAY_6101_WAVE12_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_6101_WAVE12_ATTEMPT, RELAY_6101_WAVE13_PROOF
    elif (seed, wave) == (6101, 14):
        pinned_contract, pinned_auth = RELAY_6101_WAVE13_CONTRACT, RELAY_6101_WAVE13_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE14_PROOF
    elif (seed, wave) == (6101, 15):
        pinned_contract, pinned_auth = RELAY_6101_WAVE14_CONTRACT, RELAY_6101_WAVE14_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE15_PROOF
    elif (seed, wave) == (6101, 16):
        pinned_contract, pinned_auth = RELAY_6101_WAVE15_CONTRACT, RELAY_6101_WAVE15_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE16_PROOF
    elif (seed, wave) == (6101, 17):
        pinned_contract, pinned_auth = RELAY_6101_WAVE16_CONTRACT, RELAY_6101_WAVE16_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE17_PROOF
    elif (seed, wave) == (6101, 18):
        pinned_contract, pinned_auth = RELAY_6101_WAVE17_CONTRACT, RELAY_6101_WAVE17_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_6101_WAVE17_ATTEMPT, RELAY_6101_WAVE18_PROOF
    elif (seed, wave) == (6101, 19):
        pinned_contract, pinned_auth = RELAY_6101_WAVE18_CONTRACT, RELAY_6101_WAVE18_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_6101_WAVE18_ATTEMPT, RELAY_6101_WAVE19_PROOF
    elif (seed, wave) == (6101, 20):
        pinned_contract, pinned_auth = RELAY_6101_WAVE19_CONTRACT, RELAY_6101_WAVE19_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE20_PROOF
    elif (seed, wave) == (6101, 21):
        pinned_contract, pinned_auth = RELAY_6101_WAVE20_CONTRACT, RELAY_6101_WAVE20_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE21_PROOF
    elif (seed, wave) == (6101, 22):
        pinned_contract, pinned_auth = RELAY_6101_WAVE21_CONTRACT, RELAY_6101_WAVE21_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE22_PROOF
    elif (seed, wave) == (6101, 23):
        pinned_contract, pinned_auth = RELAY_6101_WAVE22_CONTRACT, RELAY_6101_WAVE22_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE23_PROOF
    elif (seed, wave) == (6101, 24):
        pinned_contract, pinned_auth = RELAY_6101_WAVE23_CONTRACT, RELAY_6101_WAVE23_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE24_PROOF
    elif (seed, wave) == (6101, 25):
        pinned_contract, pinned_auth = RELAY_6101_WAVE24_CONTRACT, RELAY_6101_WAVE24_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE25_PROOF
    elif (seed, wave) == (6101, 26):
        pinned_contract, pinned_auth = RELAY_6101_WAVE25_CONTRACT, RELAY_6101_WAVE25_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE26_PROOF
    elif (seed, wave) == (6101, 27):
        pinned_contract, pinned_auth = RELAY_6101_WAVE26_CONTRACT, RELAY_6101_WAVE26_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE27_PROOF
    elif (seed, wave) == (6101, 28):
        pinned_contract, pinned_auth = RELAY_6101_WAVE27_CONTRACT, RELAY_6101_WAVE27_AUTH
        boundary, prior_attempt, wave_proof = 8, None, RELAY_6101_WAVE28_PROOF
    elif (seed, wave) == (2357, 3):
        pinned_contract, pinned_auth = RELAY_2357_WAVE02_CONTRACT, RELAY_2357_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_2357_WAVE02_ATTEMPT, RELAY_2357_WAVE03_PROOF
    elif (seed, wave) == (2357, 4):
        pinned_contract, pinned_auth = RELAY_2357_WAVE03_CONTRACT, RELAY_2357_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 4, None, RELAY_2357_WAVE04_PROOF
    elif (seed, wave) == (2357, 5):
        pinned_contract, pinned_auth = RELAY_2357_WAVE04_CONTRACT, RELAY_2357_WAVE04_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_2357_WAVE04_ATTEMPT, RELAY_2357_WAVE05_PROOF
    elif (seed, wave) == (2357, 6):
        pinned_contract, pinned_auth = RELAY_2357_WAVE05_CONTRACT, RELAY_2357_WAVE05_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_2357_WAVE05_ATTEMPT, RELAY_2357_WAVE06_PROOF
    elif (seed, wave) == (2357, 7):
        pinned_contract, pinned_auth = RELAY_2357_WAVE06_CONTRACT, RELAY_2357_WAVE06_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_2357_WAVE06_ATTEMPT, RELAY_2357_WAVE07_PROOF
    elif (seed, wave) == (2357, 8):
        pinned_contract, pinned_auth = RELAY_2357_WAVE07_CONTRACT, RELAY_2357_WAVE07_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE08_PROOF
    elif (seed, wave) == (2357, 9):
        pinned_contract, pinned_auth = RELAY_2357_WAVE08_CONTRACT, RELAY_2357_WAVE08_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE09_PROOF
    elif (seed, wave) == (2357, 10):
        pinned_contract, pinned_auth = RELAY_2357_WAVE09_CONTRACT, RELAY_2357_WAVE09_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_2357_WAVE09_ATTEMPT, RELAY_2357_WAVE10_PROOF
    elif (seed, wave) == (2357, 11):
        pinned_contract, pinned_auth = RELAY_2357_WAVE10_CONTRACT, RELAY_2357_WAVE10_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_2357_WAVE10_ATTEMPT, RELAY_2357_WAVE11_PROOF
    elif (seed, wave) == (2357, 12):
        pinned_contract, pinned_auth = RELAY_2357_WAVE11_CONTRACT, RELAY_2357_WAVE11_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE12_PROOF
    elif (seed, wave) == (2357, 13):
        pinned_contract, pinned_auth = RELAY_2357_WAVE12_CONTRACT, RELAY_2357_WAVE12_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE13_PROOF
    elif (seed, wave) == (2357, 14):
        pinned_contract, pinned_auth = RELAY_2357_WAVE13_CONTRACT, RELAY_2357_WAVE13_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE14_PROOF
    elif (seed, wave) == (2357, 15):
        pinned_contract, pinned_auth = RELAY_2357_WAVE14_CONTRACT, RELAY_2357_WAVE14_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE15_PROOF
    elif (seed, wave) == (2357, 16):
        pinned_contract, pinned_auth = RELAY_2357_WAVE15_CONTRACT, RELAY_2357_WAVE15_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE16_PROOF
    elif (seed, wave) == (2357, 17):
        pinned_contract, pinned_auth = RELAY_2357_WAVE16_CONTRACT, RELAY_2357_WAVE16_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE17_PROOF
    elif (seed, wave) == (2357, 18):
        pinned_contract, pinned_auth = RELAY_2357_WAVE17_CONTRACT, RELAY_2357_WAVE17_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE18_PROOF
    elif (seed, wave) == (2357, 19):
        pinned_contract, pinned_auth = RELAY_2357_WAVE18_CONTRACT, RELAY_2357_WAVE18_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE19_PROOF
    elif (seed, wave) == (2357, 20):
        pinned_contract, pinned_auth = RELAY_2357_WAVE19_CONTRACT, RELAY_2357_WAVE19_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE20_PROOF
    elif (seed, wave) == (2357, 21):
        pinned_contract, pinned_auth = RELAY_2357_WAVE20_CONTRACT, RELAY_2357_WAVE20_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE21_PROOF
    elif (seed, wave) == (2357, 22):
        pinned_contract, pinned_auth = RELAY_2357_WAVE21_CONTRACT, RELAY_2357_WAVE21_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE22_PROOF
    elif (seed, wave) == (2357, 23):
        pinned_contract, pinned_auth = RELAY_2357_WAVE22_CONTRACT, RELAY_2357_WAVE22_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE23_PROOF
    elif (seed, wave) == (2357, 24):
        pinned_contract, pinned_auth = RELAY_2357_WAVE23_CONTRACT, RELAY_2357_WAVE23_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE24_PROOF
    elif (seed, wave) == (2357, 25):
        pinned_contract, pinned_auth = RELAY_2357_WAVE24_CONTRACT, RELAY_2357_WAVE24_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE25_PROOF
    elif (seed, wave) == (2357, 26):
        pinned_contract, pinned_auth = RELAY_2357_WAVE25_CONTRACT, RELAY_2357_WAVE25_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE26_PROOF
    elif (seed, wave) == (2357, 27):
        pinned_contract, pinned_auth = RELAY_2357_WAVE26_CONTRACT, RELAY_2357_WAVE26_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE27_PROOF
    elif (seed, wave) == (2357, 28):
        pinned_contract, pinned_auth = RELAY_2357_WAVE27_CONTRACT, RELAY_2357_WAVE27_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE28_PROOF
    elif (seed, wave) == (2357, 29):
        pinned_contract, pinned_auth = RELAY_2357_WAVE28_CONTRACT, RELAY_2357_WAVE28_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE29_PROOF
    elif (seed, wave) == (2357, 30):
        pinned_contract, pinned_auth = RELAY_2357_WAVE29_CONTRACT, RELAY_2357_WAVE29_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE30_PROOF
    elif (seed, wave) == (2357, 31):
        pinned_contract, pinned_auth = RELAY_2357_WAVE30_CONTRACT, RELAY_2357_WAVE30_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_2357_WAVE31_PROOF
    elif (seed, wave) == (887, 3):
        pinned_contract, pinned_auth = RELAY_887_WAVE02_CONTRACT, RELAY_887_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_887_WAVE02_ATTEMPT, RELAY_887_WAVE03_PROOF
    elif (seed, wave) == (887, 4):
        pinned_contract, pinned_auth = RELAY_887_WAVE03_CONTRACT, RELAY_887_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_887_WAVE03_ATTEMPT, RELAY_887_WAVE04_PROOF
    elif (seed, wave) == (887, 6):
        pinned_contract, pinned_auth = RELAY_887_WAVE05_CONTRACT, RELAY_887_WAVE05_AUTH
        boundary, prior_attempt, wave_proof = 12, RELAY_887_WAVE05_ATTEMPT, RELAY_887_WAVE06_PROOF
    elif (seed, wave) == (887, 7):
        pinned_contract, pinned_auth = RELAY_887_WAVE06_CONTRACT, RELAY_887_WAVE06_AUTH
        boundary, prior_attempt, wave_proof = 12, RELAY_887_WAVE06_ATTEMPT, RELAY_887_WAVE07_PROOF
    elif (seed, wave) == (887, 8):
        pinned_contract, pinned_auth = RELAY_887_WAVE07_CONTRACT, RELAY_887_WAVE07_AUTH
        boundary, prior_attempt, wave_proof = 12, RELAY_887_WAVE07_ATTEMPT, RELAY_887_WAVE08_PROOF
    elif (seed, wave) == (887, 9):
        pinned_contract, pinned_auth = RELAY_887_WAVE08_CONTRACT, RELAY_887_WAVE08_AUTH
        boundary, prior_attempt, wave_proof = 12, RELAY_887_WAVE08_ATTEMPT, RELAY_887_WAVE09_PROOF
    elif (seed, wave) == (887, 10):
        pinned_contract, pinned_auth = RELAY_887_WAVE09_CONTRACT, RELAY_887_WAVE09_AUTH
        boundary, prior_attempt, wave_proof = 13, RELAY_887_WAVE09_ATTEMPT, RELAY_887_WAVE10_PROOF
    elif (seed, wave) == (887, 11):
        pinned_contract, pinned_auth = CONTINUATION_887_R20_CONTRACT, CONTINUATION_887_R20_AUTH
        boundary, prior_attempt, wave_proof = 32, RELAY_887_WAVE11_PREVIOUS_ATTEMPT, RELAY_887_WAVE11_PROOF
    elif (seed, wave) == (887, 12):
        pinned_contract, pinned_auth = CONTINUATION_887_R33_CONTRACT, CONTINUATION_887_R33_AUTH
        boundary, prior_attempt, wave_proof = 34, None, RELAY_887_WAVE12_PROOF
    elif (seed, wave) == (887, 13):
        pinned_contract, pinned_auth = RELAY_887_WAVE12_CONTRACT, RELAY_887_WAVE12_AUTH
        boundary, prior_attempt, wave_proof = 34, None, RELAY_887_WAVE13_PROOF
    elif (seed, wave) == (887, 14):
        pinned_contract, pinned_auth = RELAY_887_WAVE13_CONTRACT, RELAY_887_WAVE13_AUTH
        boundary, prior_attempt, wave_proof = 34, None, RELAY_887_WAVE14_PROOF
    elif (seed, wave) == (887, 15):
        pinned_contract, pinned_auth = RELAY_887_WAVE14_CONTRACT, RELAY_887_WAVE14_AUTH
        boundary, prior_attempt, wave_proof = 34, None, RELAY_887_WAVE15_PROOF
    elif (seed, wave) == (157, 4):
        pinned_contract, pinned_auth = RELAY_157_WAVE03_CONTRACT, RELAY_157_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 3, RELAY_157_WAVE03_ATTEMPT, RELAY_157_WAVE04_PROOF
    elif (seed, wave) == (12203, 4):
        pinned_contract, pinned_auth = RELAY_12203_WAVE03_CONTRACT, RELAY_12203_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_12203_WAVE03_ATTEMPT, RELAY_12203_WAVE04_PROOF
    elif (seed, wave) == (157, 5):
        pinned_contract, pinned_auth = RELAY_157_WAVE04_CONTRACT, RELAY_157_WAVE04_AUTH
        boundary, prior_attempt, wave_proof = 3, RELAY_157_WAVE04_ATTEMPT, RELAY_157_WAVE05_PROOF
    elif (seed, wave) == (12203, 5):
        pinned_contract, pinned_auth = RELAY_12203_WAVE04_CONTRACT, RELAY_12203_WAVE04_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_12203_WAVE04_ATTEMPT, RELAY_12203_WAVE05_PROOF
    elif (seed, wave) == (157, 6):
        pinned_contract, pinned_auth = RELAY_157_WAVE05_CONTRACT, RELAY_157_WAVE05_AUTH
        boundary, prior_attempt, wave_proof = 3, RELAY_157_WAVE05_ATTEMPT, RELAY_157_WAVE06_PROOF
    elif (seed, wave) == (157, 8):
        pinned_contract, pinned_auth = RELAY_157_WAVE07_CONTRACT, RELAY_157_WAVE07_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_157_WAVE07_ATTEMPT, RELAY_157_WAVE08_PROOF
    elif (seed, wave) == (157, 9):
        pinned_contract, pinned_auth = RELAY_157_WAVE08_CONTRACT, RELAY_157_WAVE08_AUTH
        boundary, prior_attempt, wave_proof = 7, RELAY_157_WAVE08_ATTEMPT, RELAY_157_WAVE09_PROOF
    elif (seed, wave) == (157, 10):
        pinned_contract, pinned_auth = RELAY_157_WAVE09_CONTRACT, RELAY_157_WAVE09_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE10_PROOF
    elif (seed, wave) == (157, 11):
        pinned_contract, pinned_auth = RELAY_157_WAVE10_CONTRACT, RELAY_157_WAVE10_AUTH
        boundary, prior_attempt, wave_proof = 7, RELAY_157_WAVE10_ATTEMPT, RELAY_157_WAVE11_PROOF
    elif (seed, wave) == (157, 12):
        pinned_contract, pinned_auth = RELAY_157_WAVE11_CONTRACT, RELAY_157_WAVE11_AUTH
        boundary, prior_attempt, wave_proof = 7, RELAY_157_WAVE11_ATTEMPT, RELAY_157_WAVE12_PROOF
    elif (seed, wave) == (157, 13):
        pinned_contract, pinned_auth = RELAY_157_WAVE12_CONTRACT, RELAY_157_WAVE12_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE13_PROOF
    elif (seed, wave) == (157, 14):
        pinned_contract, pinned_auth = RELAY_157_WAVE13_CONTRACT, RELAY_157_WAVE13_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE14_PROOF
    elif (seed, wave) == (157, 15):
        pinned_contract, pinned_auth = RELAY_157_WAVE14_CONTRACT, RELAY_157_WAVE14_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE15_PROOF
    elif (seed, wave) == (157, 16):
        pinned_contract, pinned_auth = RELAY_157_WAVE15_CONTRACT, RELAY_157_WAVE15_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE16_PROOF
    elif (seed, wave) == (157, 17):
        pinned_contract, pinned_auth = RELAY_157_WAVE16_CONTRACT, RELAY_157_WAVE16_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE17_PROOF
    elif (seed, wave) == (157, 18):
        pinned_contract, pinned_auth = RELAY_157_WAVE17_CONTRACT, RELAY_157_WAVE17_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE18_PROOF
    elif (seed, wave) == (157, 19):
        pinned_contract, pinned_auth = RELAY_157_WAVE18_CONTRACT, RELAY_157_WAVE18_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE19_PROOF
    elif (seed, wave) == (157, 20):
        pinned_contract, pinned_auth = RELAY_157_WAVE19_CONTRACT, RELAY_157_WAVE19_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE20_PROOF
    elif (seed, wave) == (157, 21):
        pinned_contract, pinned_auth = RELAY_157_WAVE20_CONTRACT, RELAY_157_WAVE20_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE21_PROOF
    elif (seed, wave) == (157, 22):
        pinned_contract, pinned_auth = RELAY_157_WAVE21_CONTRACT, RELAY_157_WAVE21_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE22_PROOF
    elif (seed, wave) == (157, 23):
        pinned_contract, pinned_auth = RELAY_157_WAVE22_CONTRACT, RELAY_157_WAVE22_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE23_PROOF
    elif (seed, wave) == (157, 24):
        pinned_contract, pinned_auth = RELAY_157_WAVE23_CONTRACT, RELAY_157_WAVE23_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE24_PROOF
    elif (seed, wave) == (157, 25):
        pinned_contract, pinned_auth = RELAY_157_WAVE24_CONTRACT, RELAY_157_WAVE24_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE25_PROOF
    elif (seed, wave) == (157, 26):
        pinned_contract, pinned_auth = RELAY_157_WAVE25_CONTRACT, RELAY_157_WAVE25_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE26_PROOF
    elif (seed, wave) == (157, 27):
        pinned_contract, pinned_auth = RELAY_157_WAVE26_CONTRACT, RELAY_157_WAVE26_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE27_PROOF
    elif (seed, wave) == (157, 28):
        pinned_contract, pinned_auth = RELAY_157_WAVE27_CONTRACT, RELAY_157_WAVE27_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE28_PROOF
    elif (seed, wave) == (157, 29):
        pinned_contract, pinned_auth = RELAY_157_WAVE28_CONTRACT, RELAY_157_WAVE28_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE29_PROOF
    elif (seed, wave) == (157, 30):
        pinned_contract, pinned_auth = RELAY_157_WAVE29_CONTRACT, RELAY_157_WAVE29_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE30_PROOF
    elif (seed, wave) == (157, 31):
        pinned_contract, pinned_auth = RELAY_157_WAVE30_CONTRACT, RELAY_157_WAVE30_AUTH
        boundary, prior_attempt, wave_proof = 7, None, RELAY_157_WAVE31_PROOF
    elif (seed, wave) == (12203, 6):
        pinned_contract, pinned_auth = RELAY_12203_WAVE05_CONTRACT, RELAY_12203_WAVE05_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_12203_WAVE05_ATTEMPT, RELAY_12203_WAVE06_PROOF
    elif (seed, wave) == (12203, 8):
        pinned_contract, pinned_auth = RELAY_12203_WAVE07_CONTRACT, RELAY_12203_WAVE07_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE08_PROOF
    elif (seed, wave) == (12203, 9):
        pinned_contract, pinned_auth = RELAY_12203_WAVE08_CONTRACT, RELAY_12203_WAVE08_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE09_PROOF
    elif (seed, wave) == (12203, 10):
        pinned_contract, pinned_auth = RELAY_12203_WAVE09_CONTRACT, RELAY_12203_WAVE09_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE09_ATTEMPT, RELAY_12203_WAVE10_PROOF
    elif (seed, wave) == (12203, 11):
        pinned_contract, pinned_auth = RELAY_12203_WAVE10_CONTRACT, RELAY_12203_WAVE10_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE10_ATTEMPT, RELAY_12203_WAVE11_PROOF
    elif (seed, wave) == (12203, 12):
        pinned_contract, pinned_auth = RELAY_12203_WAVE11_CONTRACT, RELAY_12203_WAVE11_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE12_PROOF
    elif (seed, wave) == (12203, 13):
        pinned_contract, pinned_auth = RELAY_12203_WAVE12_CONTRACT, RELAY_12203_WAVE12_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE13_PROOF
    elif (seed, wave) == (12203, 14):
        pinned_contract, pinned_auth = RELAY_12203_WAVE13_CONTRACT, RELAY_12203_WAVE13_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE13_ATTEMPT, RELAY_12203_WAVE14_PROOF
    elif (seed, wave) == (12203, 15):
        pinned_contract, pinned_auth = RELAY_12203_WAVE14_CONTRACT, RELAY_12203_WAVE14_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE14_ATTEMPT, RELAY_12203_WAVE15_PROOF
    elif (seed, wave) == (12203, 16):
        pinned_contract, pinned_auth = RELAY_12203_WAVE15_CONTRACT, RELAY_12203_WAVE15_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE15_ATTEMPT, RELAY_12203_WAVE16_PROOF
    elif (seed, wave) == (12203, 17):
        pinned_contract, pinned_auth = RELAY_12203_WAVE16_CONTRACT, RELAY_12203_WAVE16_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE16_ATTEMPT, RELAY_12203_WAVE17_PROOF
    elif (seed, wave) == (12203, 18):
        pinned_contract, pinned_auth = RELAY_12203_WAVE17_CONTRACT, RELAY_12203_WAVE17_AUTH
        boundary, prior_attempt, wave_proof = 5, RELAY_12203_WAVE17_ATTEMPT, RELAY_12203_WAVE18_PROOF
    elif (seed, wave) == (12203, 19):
        pinned_contract, pinned_auth = RELAY_12203_WAVE18_CONTRACT, RELAY_12203_WAVE18_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE19_PROOF
    elif (seed, wave) == (12203, 20):
        pinned_contract, pinned_auth = RELAY_12203_WAVE19_CONTRACT, RELAY_12203_WAVE19_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE20_PROOF
    elif (seed, wave) == (12203, 21):
        pinned_contract, pinned_auth = RELAY_12203_WAVE20_CONTRACT, RELAY_12203_WAVE20_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE21_PROOF
    elif (seed, wave) == (12203, 22):
        pinned_contract, pinned_auth = RELAY_12203_WAVE21_CONTRACT, RELAY_12203_WAVE21_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE22_PROOF
    elif (seed, wave) == (12203, 23):
        pinned_contract, pinned_auth = RELAY_12203_WAVE22_CONTRACT, RELAY_12203_WAVE22_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE23_PROOF
    elif (seed, wave) == (12203, 24):
        pinned_contract, pinned_auth = RELAY_12203_WAVE23_CONTRACT, RELAY_12203_WAVE23_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE24_PROOF
    elif (seed, wave) == (12203, 25):
        pinned_contract, pinned_auth = RELAY_12203_WAVE24_CONTRACT, RELAY_12203_WAVE24_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE25_PROOF
    elif (seed, wave) == (12203, 26):
        pinned_contract, pinned_auth = RELAY_12203_WAVE25_CONTRACT, RELAY_12203_WAVE25_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE26_PROOF
    elif (seed, wave) == (12203, 27):
        pinned_contract, pinned_auth = RELAY_12203_WAVE26_CONTRACT, RELAY_12203_WAVE26_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE27_PROOF
    elif (seed, wave) == (12203, 28):
        pinned_contract, pinned_auth = RELAY_12203_WAVE27_CONTRACT, RELAY_12203_WAVE27_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE28_PROOF
    elif (seed, wave) == (12203, 29):
        pinned_contract, pinned_auth = RELAY_12203_WAVE28_CONTRACT, RELAY_12203_WAVE28_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE29_PROOF
    elif (seed, wave) == (12203, 30):
        pinned_contract, pinned_auth = RELAY_12203_WAVE29_CONTRACT, RELAY_12203_WAVE29_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE30_PROOF
    elif (seed, wave) == (12203, 31):
        pinned_contract, pinned_auth = RELAY_12203_WAVE30_CONTRACT, RELAY_12203_WAVE30_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE31_PROOF
    elif (seed, wave) == (12203, 32):
        pinned_contract, pinned_auth = RELAY_12203_WAVE31_CONTRACT, RELAY_12203_WAVE31_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE32_PROOF
    elif (seed, wave) == (12203, 33):
        pinned_contract, pinned_auth = RELAY_12203_WAVE32_CONTRACT, RELAY_12203_WAVE32_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE33_PROOF
    elif (seed, wave) == (12203, 34):
        pinned_contract, pinned_auth = RELAY_12203_WAVE33_CONTRACT, RELAY_12203_WAVE33_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE34_PROOF
    elif (seed, wave) == (12203, 35):
        pinned_contract, pinned_auth = RELAY_12203_WAVE34_CONTRACT, RELAY_12203_WAVE34_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE35_PROOF
    elif (seed, wave) == (12203, 36):
        pinned_contract, pinned_auth = RELAY_12203_WAVE35_CONTRACT, RELAY_12203_WAVE35_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE36_PROOF
    elif (seed, wave) == (12203, 37):
        pinned_contract, pinned_auth = RELAY_12203_WAVE36_CONTRACT, RELAY_12203_WAVE36_AUTH
        boundary, prior_attempt, wave_proof = 5, None, RELAY_12203_WAVE37_PROOF
    else:
        return None
    if previous_contract != pinned_contract:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-{wave:02d}"
    physical_attempts = wave_proof["newly_authorized_boundary_physical_attempts"]
    previous_suffix = {(6101, 9): "r5-continuation-01", (887, 11): "r20-continuation-01",
                       (887, 12): "r33-continuation-01"}.get(
        (seed, wave), f"relay-retry-{wave - 1:02d}",
    )
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor",
                                           physical_attempts, previous_suffix=previous_suffix)
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    registered = {(2357, 5): (RELAY_2357_WAVE05_CONTRACT, RELAY_2357_WAVE05_AUTH),
                  (2357, 6): (RELAY_2357_WAVE06_CONTRACT, RELAY_2357_WAVE06_AUTH),
                  (2357, 7): (RELAY_2357_WAVE07_CONTRACT, RELAY_2357_WAVE07_AUTH),
                  (2357, 8): (RELAY_2357_WAVE08_CONTRACT, RELAY_2357_WAVE08_AUTH),
                  (2357, 9): (RELAY_2357_WAVE09_CONTRACT, RELAY_2357_WAVE09_AUTH),
                  (2357, 10): (RELAY_2357_WAVE10_CONTRACT, RELAY_2357_WAVE10_AUTH),
                  (2357, 11): (RELAY_2357_WAVE11_CONTRACT, RELAY_2357_WAVE11_AUTH),
                  (2357, 12): (RELAY_2357_WAVE12_CONTRACT, RELAY_2357_WAVE12_AUTH),
                  (2357, 13): (RELAY_2357_WAVE13_CONTRACT, RELAY_2357_WAVE13_AUTH),
                  (2357, 14): (RELAY_2357_WAVE14_CONTRACT, RELAY_2357_WAVE14_AUTH),
                  (2357, 15): (RELAY_2357_WAVE15_CONTRACT, RELAY_2357_WAVE15_AUTH),
                  (2357, 16): (RELAY_2357_WAVE16_CONTRACT, RELAY_2357_WAVE16_AUTH),
                  (2357, 17): (RELAY_2357_WAVE17_CONTRACT, RELAY_2357_WAVE17_AUTH),
                  (2357, 18): (RELAY_2357_WAVE18_CONTRACT, RELAY_2357_WAVE18_AUTH),
                  (2357, 19): (RELAY_2357_WAVE19_CONTRACT, RELAY_2357_WAVE19_AUTH),
                  (2357, 20): (RELAY_2357_WAVE20_CONTRACT, RELAY_2357_WAVE20_AUTH),
                  (2357, 21): (RELAY_2357_WAVE21_CONTRACT, RELAY_2357_WAVE21_AUTH),
                  (2357, 22): (RELAY_2357_WAVE22_CONTRACT, RELAY_2357_WAVE22_AUTH),
                  (2357, 23): (RELAY_2357_WAVE23_CONTRACT, RELAY_2357_WAVE23_AUTH),
                  (2357, 24): (RELAY_2357_WAVE24_CONTRACT, RELAY_2357_WAVE24_AUTH),
                  (2357, 25): (RELAY_2357_WAVE25_CONTRACT, RELAY_2357_WAVE25_AUTH),
                  (2357, 26): (RELAY_2357_WAVE26_CONTRACT, RELAY_2357_WAVE26_AUTH),
                  (2357, 27): (RELAY_2357_WAVE27_CONTRACT, RELAY_2357_WAVE27_AUTH),
                  (2357, 28): (RELAY_2357_WAVE28_CONTRACT, RELAY_2357_WAVE28_AUTH),
                  (2357, 29): (RELAY_2357_WAVE29_CONTRACT, RELAY_2357_WAVE29_AUTH),
                  (2357, 30): (RELAY_2357_WAVE30_CONTRACT, RELAY_2357_WAVE30_AUTH),
                  (2357, 31): (RELAY_2357_WAVE31_CONTRACT, RELAY_2357_WAVE31_AUTH),
                  (6101, 6): (RELAY_6101_WAVE06_CONTRACT, RELAY_6101_WAVE06_AUTH),
                  (6101, 8): (RELAY_6101_WAVE08_CONTRACT, RELAY_6101_WAVE08_AUTH),
                  (6101, 9): (RELAY_6101_WAVE09_CONTRACT, RELAY_6101_WAVE09_AUTH),
                  (6101, 10): (RELAY_6101_WAVE10_CONTRACT, RELAY_6101_WAVE10_AUTH),
                  (6101, 11): (RELAY_6101_WAVE11_CONTRACT, RELAY_6101_WAVE11_AUTH),
                  (6101, 12): (RELAY_6101_WAVE12_CONTRACT, RELAY_6101_WAVE12_AUTH),
                  (6101, 13): (RELAY_6101_WAVE13_CONTRACT, RELAY_6101_WAVE13_AUTH),
                  (6101, 14): (RELAY_6101_WAVE14_CONTRACT, RELAY_6101_WAVE14_AUTH),
                  (6101, 15): (RELAY_6101_WAVE15_CONTRACT, RELAY_6101_WAVE15_AUTH),
                  (6101, 16): (RELAY_6101_WAVE16_CONTRACT, RELAY_6101_WAVE16_AUTH),
                  (6101, 17): (RELAY_6101_WAVE17_CONTRACT, RELAY_6101_WAVE17_AUTH),
                  (6101, 18): (RELAY_6101_WAVE18_CONTRACT, RELAY_6101_WAVE18_AUTH),
                  (6101, 19): (RELAY_6101_WAVE19_CONTRACT, RELAY_6101_WAVE19_AUTH),
                  (6101, 20): (RELAY_6101_WAVE20_CONTRACT, RELAY_6101_WAVE20_AUTH),
                  (6101, 21): (RELAY_6101_WAVE21_CONTRACT, RELAY_6101_WAVE21_AUTH),
                  (6101, 22): (RELAY_6101_WAVE22_CONTRACT, RELAY_6101_WAVE22_AUTH),
                  (6101, 23): (RELAY_6101_WAVE23_CONTRACT, RELAY_6101_WAVE23_AUTH),
                  (6101, 24): (RELAY_6101_WAVE24_CONTRACT, RELAY_6101_WAVE24_AUTH),
                  (6101, 25): (RELAY_6101_WAVE25_CONTRACT, RELAY_6101_WAVE25_AUTH),
                  (6101, 26): (RELAY_6101_WAVE26_CONTRACT, RELAY_6101_WAVE26_AUTH),
                  (6101, 27): (RELAY_6101_WAVE27_CONTRACT, RELAY_6101_WAVE27_AUTH),
                  (6101, 28): (RELAY_6101_WAVE28_CONTRACT, RELAY_6101_WAVE28_AUTH),
                  (887, 6): (RELAY_887_WAVE06_CONTRACT, RELAY_887_WAVE06_AUTH),
                  (887, 7): (RELAY_887_WAVE07_CONTRACT, RELAY_887_WAVE07_AUTH),
                  (887, 8): (RELAY_887_WAVE08_CONTRACT, RELAY_887_WAVE08_AUTH),
                  (887, 9): (RELAY_887_WAVE09_CONTRACT, RELAY_887_WAVE09_AUTH),
                  (887, 10): (RELAY_887_WAVE10_CONTRACT, RELAY_887_WAVE10_AUTH),
                  (887, 11): (RELAY_887_WAVE11_CONTRACT, RELAY_887_WAVE11_AUTH),
                  (887, 12): (RELAY_887_WAVE12_CONTRACT, RELAY_887_WAVE12_AUTH),
                  (887, 13): (RELAY_887_WAVE13_CONTRACT, RELAY_887_WAVE13_AUTH),
                  (887, 14): (RELAY_887_WAVE14_CONTRACT, RELAY_887_WAVE14_AUTH),
                  (887, 15): (RELAY_887_WAVE15_CONTRACT, RELAY_887_WAVE15_AUTH),
                  (157, 8): (RELAY_157_WAVE08_CONTRACT, RELAY_157_WAVE08_AUTH),
                  (157, 9): (RELAY_157_WAVE09_CONTRACT, RELAY_157_WAVE09_AUTH),
                  (157, 10): (RELAY_157_WAVE10_CONTRACT, RELAY_157_WAVE10_AUTH),
                  (157, 11): (RELAY_157_WAVE11_CONTRACT, RELAY_157_WAVE11_AUTH),
                  (157, 12): (RELAY_157_WAVE12_CONTRACT, RELAY_157_WAVE12_AUTH),
                  (157, 13): (RELAY_157_WAVE13_CONTRACT, RELAY_157_WAVE13_AUTH),
                  (157, 14): (RELAY_157_WAVE14_CONTRACT, RELAY_157_WAVE14_AUTH),
                  (157, 15): (RELAY_157_WAVE15_CONTRACT, RELAY_157_WAVE15_AUTH),
                  (157, 16): (RELAY_157_WAVE16_CONTRACT, RELAY_157_WAVE16_AUTH),
                  (157, 17): (RELAY_157_WAVE17_CONTRACT, RELAY_157_WAVE17_AUTH),
                  (157, 18): (RELAY_157_WAVE18_CONTRACT, RELAY_157_WAVE18_AUTH),
                  (157, 19): (RELAY_157_WAVE19_CONTRACT, RELAY_157_WAVE19_AUTH),
                  (157, 20): (RELAY_157_WAVE20_CONTRACT, RELAY_157_WAVE20_AUTH),
                  (157, 21): (RELAY_157_WAVE21_CONTRACT, RELAY_157_WAVE21_AUTH),
                  (157, 22): (RELAY_157_WAVE22_CONTRACT, RELAY_157_WAVE22_AUTH),
                  (157, 23): (RELAY_157_WAVE23_CONTRACT, RELAY_157_WAVE23_AUTH),
                  (157, 24): (RELAY_157_WAVE24_CONTRACT, RELAY_157_WAVE24_AUTH),
                  (157, 25): (RELAY_157_WAVE25_CONTRACT, RELAY_157_WAVE25_AUTH),
                  (157, 26): (RELAY_157_WAVE26_CONTRACT, RELAY_157_WAVE26_AUTH),
                  (157, 27): (RELAY_157_WAVE27_CONTRACT, RELAY_157_WAVE27_AUTH),
                  (157, 28): (RELAY_157_WAVE28_CONTRACT, RELAY_157_WAVE28_AUTH),
                  (157, 29): (RELAY_157_WAVE29_CONTRACT, RELAY_157_WAVE29_AUTH),
                  (157, 30): (RELAY_157_WAVE30_CONTRACT, RELAY_157_WAVE30_AUTH),
                  (157, 31): (RELAY_157_WAVE31_CONTRACT, RELAY_157_WAVE31_AUTH),
                  (12203, 9): (RELAY_12203_WAVE09_CONTRACT, RELAY_12203_WAVE09_AUTH),
                  (12203, 10): (RELAY_12203_WAVE10_CONTRACT, RELAY_12203_WAVE10_AUTH),
                  (12203, 11): (RELAY_12203_WAVE11_CONTRACT, RELAY_12203_WAVE11_AUTH),
                  (12203, 12): (RELAY_12203_WAVE12_CONTRACT, RELAY_12203_WAVE12_AUTH),
                  (12203, 13): (RELAY_12203_WAVE13_CONTRACT, RELAY_12203_WAVE13_AUTH),
                  (12203, 14): (RELAY_12203_WAVE14_CONTRACT, RELAY_12203_WAVE14_AUTH),
                  (12203, 15): (RELAY_12203_WAVE15_CONTRACT, RELAY_12203_WAVE15_AUTH),
                  (12203, 16): (RELAY_12203_WAVE16_CONTRACT, RELAY_12203_WAVE16_AUTH),
                  (12203, 17): (RELAY_12203_WAVE17_CONTRACT, RELAY_12203_WAVE17_AUTH),
                  (12203, 18): (RELAY_12203_WAVE18_CONTRACT, RELAY_12203_WAVE18_AUTH),
                  (12203, 19): (RELAY_12203_WAVE19_CONTRACT, RELAY_12203_WAVE19_AUTH),
                  (12203, 20): (RELAY_12203_WAVE20_CONTRACT, RELAY_12203_WAVE20_AUTH),
                  (12203, 21): (RELAY_12203_WAVE21_CONTRACT, RELAY_12203_WAVE21_AUTH),
                  (12203, 22): (RELAY_12203_WAVE22_CONTRACT, RELAY_12203_WAVE22_AUTH),
                  (12203, 23): (RELAY_12203_WAVE23_CONTRACT, RELAY_12203_WAVE23_AUTH),
                  (12203, 24): (RELAY_12203_WAVE24_CONTRACT, RELAY_12203_WAVE24_AUTH),
                  (12203, 25): (RELAY_12203_WAVE25_CONTRACT, RELAY_12203_WAVE25_AUTH),
                  (12203, 26): (RELAY_12203_WAVE26_CONTRACT, RELAY_12203_WAVE26_AUTH),
                  (12203, 27): (RELAY_12203_WAVE27_CONTRACT, RELAY_12203_WAVE27_AUTH),
                  (12203, 28): (RELAY_12203_WAVE28_CONTRACT, RELAY_12203_WAVE28_AUTH),
                  (12203, 29): (RELAY_12203_WAVE29_CONTRACT, RELAY_12203_WAVE29_AUTH),
                  (12203, 30): (RELAY_12203_WAVE30_CONTRACT, RELAY_12203_WAVE30_AUTH),
                  (12203, 31): (RELAY_12203_WAVE31_CONTRACT, RELAY_12203_WAVE31_AUTH),
                  (12203, 32): (RELAY_12203_WAVE32_CONTRACT, RELAY_12203_WAVE32_AUTH),
                  (12203, 33): (RELAY_12203_WAVE33_CONTRACT, RELAY_12203_WAVE33_AUTH),
                  (12203, 34): (RELAY_12203_WAVE34_CONTRACT, RELAY_12203_WAVE34_AUTH),
                  (12203, 35): (RELAY_12203_WAVE35_CONTRACT, RELAY_12203_WAVE35_AUTH),
                  (12203, 36): (RELAY_12203_WAVE36_CONTRACT, RELAY_12203_WAVE36_AUTH),
                  (12203, 37): (RELAY_12203_WAVE37_CONTRACT, RELAY_12203_WAVE37_AUTH)}.get((seed, wave))
    if registered is not None and (contract, record_digest(read_json(run / "authorization.json"))) != registered:
        return None
    signed = read_json(run / "formal_contract.json")
    previous_record = read_json(previous / "formal_contract.json")
    previous_proof = previous_record.get("predecessor") or {}
    previous_profile = previous_record.get("profile", {})
    fitted = {(157, 9): (4, RELAY_157_WAVE09_FITTED_FILES, RELAY_157_WAVE09_CHECKPOINT),
              (2357, 8): (4, RELAY_2357_WAVE08_FITTED_FILES, RELAY_2357_WAVE08_CHECKPOINT),
              (6101, 9): (5, RELAY_6101_WAVE09_FITTED_FILES, RELAY_6101_WAVE09_CHECKPOINT),
              (6101, 10): (6, RELAY_6101_WAVE10_FITTED_FILES, RELAY_6101_WAVE10_CHECKPOINT),
              (887, 10): (12, RELAY_887_WAVE10_FITTED_FILES, RELAY_887_WAVE10_CHECKPOINT),
              (887, 11): (20, RELAY_887_WAVE11_FITTED_FILES, RELAY_887_WAVE11_CHECKPOINT),
              (887, 12): (33, RELAY_887_WAVE12_FITTED_FILES, RELAY_887_WAVE12_CHECKPOINT)}.get((seed, wave))
    directory = previous / f"seed_{seed}"
    if fitted is not None:
        old_start, fitted_files, checkpoint = fitted
        old_boundary = previous_profile.get("resume_boundaries", {}).get(str(seed), {})
        boundaries = profile.get("resume_boundaries", {})
        descriptor = boundaries.get(str(seed), {})
        if (old_boundary.get("round") != old_start or set(boundaries) != {str(seed)}
                or set(descriptor) != {"round", "state_sha256", "memory_sha256", "user_message_sha256"}
                or any(not isinstance(value, str) or not SHA.fullmatch(value)
                       for key, value in descriptor.items() if key != "round")
                or latest_round(directory)[1].get("checkpoint_sha256") != checkpoint
                or any(checksum(directory / name) != sha for name, sha in fitted_files.items())):
            return None
        previous_profile = {**previous_profile, "resume_boundaries": boundaries}
    if (seed, wave) == (887, 12):
        if (previous_profile.get("full_imported_boundary_round") != 33
                or previous_profile.get("replay_eligible_from_round") != 33
                or descriptor != RELAY_887_WAVE12_DESCRIPTOR
                or (directory / "round_35").exists() or not (directory / "pause.request").is_file()):
            return None
        previous_profile = {name: value for name, value in previous_profile.items() if name != "full_imported_boundary_round"}
    post_r34_retry = seed == 887 and 13 <= wave <= 15
    if post_r34_retry:
        fitted_files, prior_prefix = {
            13: (RELAY_887_WAVE13_FITTED_FILES, RELAY_887_WAVE13_PREFIX_SHA),
            14: (RELAY_887_WAVE14_FITTED_FILES, RELAY_887_WAVE14_PREFIX_SHA),
            15: (RELAY_887_WAVE15_FITTED_FILES, RELAY_887_WAVE15_PREFIX_SHA),
        }[wave]
        if (profile.get("resume_boundaries", {}).get("887") != RELAY_887_WAVE12_DESCRIPTOR
                or previous_proof.get("checkpoint_sha256") != RELAY_887_WAVE12_CHECKPOINT
                or proof.get("previous_prefix_sha256") != prior_prefix
                or any(checksum(directory / name) != sha for name, sha in fitted_files.items())
                or (directory / "round_35").exists() or not (directory / "pause.request").is_file()):
            return None
    replay_round = start + 1
    if physical_attempts == 2:
        replay_round = start
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    elif (seed, wave) == (6101, 9):
        if (previous_profile.get("replay_eligible_from_round") != old_start
                or (directory / "round_05/boundary_user_message.json").exists()):
            return None
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    elif (seed, wave) == (887, 11):
        if (previous_profile.get("replay_eligible_from_round") != old_start
                or descriptor != RELAY_887_WAVE11_DESCRIPTOR
                or (directory / "round_20/boundary_user_message.json").exists()
                or (directory / "round_33").exists()
                or not (directory / "pause.request").is_file()):
            return None
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    elif (seed, wave) == (12203, 13):
        if previous_profile.get("replay_eligible_from_round") != start:
            return None
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    elif (seed, wave) in ((6101, 5), (6101, 18), (2357, 5), (2357, 7), (2357, 10), (887, 7), (887, 9), (157, 11),
                          (887, 10), (6101, 12),
                          (12203, 10), (12203, 15), (12203, 17)):
        if (previous_profile.get("replay_eligible_from_round") != start
                or (previous / f"seed_{seed}/round_{start:02d}/turn_00/attempt_01.started.json").exists()):
            return None
        if (seed, wave) == (6101, 5) and (contract, record_digest(read_json(run / "authorization.json"))) != (
            RELAY_6101_WAVE05_CONTRACT, RELAY_6101_WAVE05_AUTH,
        ):
            return None
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    exact = {**wave_proof, "previous_authorization_sha256": pinned_auth}
    if (seed, wave) == (887, 12):
        exact["reviewed_normal_parent_budget"] = {"round": 33, "recovery_budget": CONTINUATION_887_R33_SCOPE,
                                                   "execution_budget": CONTINUATION_887_R33_BUDGET}
    if post_r34_retry:
        exact.update(reviewed_fitted_prefix_files=fitted_files,
                     checkpoint_sha256=RELAY_887_WAVE12_CHECKPOINT,
                     reviewed_normal_parent_budget=previous_proof.get("reviewed_normal_parent_budget"))
    elif seed == 887 and wave in (6, 7, 8, 9):
        exact.update(reviewed_fitted_prefix_files=RELAY_887_R12_FITTED_FILES, checkpoint_sha256=RELAY_887_R12_CHECKPOINT)
    elif seed == 887 and fitted is None:
        if "cached_import" in proof:
            return None
        exact["reviewed_cached_import"] = RELAY_887_CACHED_REVIEW
    if seed == 157 and 8 <= wave <= 31:
        if "cached_import" in proof or previous_proof.get("reviewed_cached_import") != RELAY_157_WAVE07_CACHED_REVIEW:
            return None
        exact.update(reviewed_cached_import=RELAY_157_WAVE07_CACHED_REVIEW,
                     public_resource_receipt_sha256=RELAY_157_WAVE07_RESOURCE_SHA)
        if wave == 8:
            exact["reviewed_fitted_prefix_files"] = CACHED_R3_157_FITTED_FILES
    if fitted is not None:
        exact.update(reviewed_fitted_prefix_files=fitted_files, checkpoint_sha256=checkpoint)
    inherited_fit = {
        (157, 10): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 11): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 12): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 13): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 14): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 15): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 16): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 17): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 18): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 19): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 20): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 21): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 22): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 23): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 24): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 25): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 26): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 27): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 28): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 29): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 30): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (157, 31): (RELAY_157_WAVE09_CHECKPOINT, RELAY_157_WAVE09_FITTED_FILES),
        (2357, 9): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 10): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 11): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 12): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 13): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 14): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 15): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 16): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 17): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 18): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 19): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 20): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 21): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 22): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 23): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 24): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 25): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 26): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 27): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 28): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 29): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 30): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (2357, 31): (RELAY_2357_WAVE08_CHECKPOINT, RELAY_2357_WAVE08_FITTED_FILES),
        (6101, 11): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 12): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 13): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 14): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 15): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 16): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 17): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 18): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 19): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 20): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 21): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 22): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 23): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 24): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 25): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 26): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 27): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (6101, 28): (RELAY_6101_WAVE10_CHECKPOINT, RELAY_6101_WAVE10_FITTED_FILES),
        (887, 8): (RELAY_887_R12_CHECKPOINT, RELAY_887_R12_FITTED_FILES),
        (887, 9): (RELAY_887_R12_CHECKPOINT, RELAY_887_R12_FITTED_FILES),
    }.get((seed, wave))
    if inherited_fit is not None:
        checkpoint, fitted_files = inherited_fit
        if (previous_proof.get("checkpoint_sha256"), previous_proof.get("reviewed_fitted_prefix_files")) != inherited_fit:
            return None
        exact.update(checkpoint_sha256=checkpoint, reviewed_fitted_prefix_files=fitted_files)
    if seed == 12203 and 15 <= wave <= 37:
        if previous_proof.get("checkpoint_sha256") != RELAY_12203_WAVE15_CHECKPOINT:
            return None
        exact["checkpoint_sha256"] = RELAY_12203_WAVE15_CHECKPOINT
    if (seed, wave) == (6101, 8):
        if "cached_import" in proof or previous_proof.get("reviewed_cached_import") != RELAY_6101_WAVE07_CACHED_REVIEW:
            return None
        exact.update(reviewed_cached_import=RELAY_6101_WAVE07_CACHED_REVIEW,
                     reviewed_fitted_prefix_files=RELAY_6101_WAVE07_FITTED_FILES,
                     public_resource_receipt_sha256=RELAY_6101_WAVE07_RESOURCE_SHA,
                     checkpoint_sha256=CACHED_R2_6101_CHECKPOINT)
    if seed == 2357 and wave in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        if type(profile.get("response_headers_seconds")) is not int or profile["response_headers_seconds"] != 300:
            return None
        if wave == 19:
            previous_profile = {**previous_profile, "response_headers_seconds": 300}
    if (start != boundary or profile != previous_profile
            or profile.get("replay_eligible_from_round") != replay_round
            or profile.get("audited_future_transport_replay") is not True
            or type(profile.get("sealed_relay_retry_seconds")) is not int
            or profile["sealed_relay_retry_seconds"] != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    checked_budgets = {
        (2357, 5): (2, RELAY_2357_WAVE04_PROOF["execution_budget"]),
        (2357, 6): (1, RELAY_2357_WAVE05_PROOF["execution_budget"]),
        (2357, 7): (2, RELAY_2357_WAVE06_PROOF["execution_budget"]),
        (2357, 8): (1, RELAY_2357_WAVE07_PROOF["execution_budget"]),
        (2357, 9): (2, RELAY_2357_WAVE08_PROOF["execution_budget"]),
        (2357, 10): (2, RELAY_2357_WAVE09_PROOF["execution_budget"]),
        (2357, 11): (1, RELAY_2357_WAVE10_PROOF["execution_budget"]),
        (2357, 12): (2, RELAY_2357_WAVE11_PROOF["execution_budget"]),
        (2357, 13): (2, RELAY_2357_WAVE12_PROOF["execution_budget"]),
        (2357, 14): (2, RELAY_2357_WAVE13_PROOF["execution_budget"]),
        (2357, 15): (2, RELAY_2357_WAVE14_PROOF["execution_budget"]),
        (2357, 16): (2, RELAY_2357_WAVE15_PROOF["execution_budget"]),
        (2357, 17): (2, RELAY_2357_WAVE16_PROOF["execution_budget"]),
        (2357, 18): (2, RELAY_2357_WAVE17_PROOF["execution_budget"]),
        (2357, 19): (2, RELAY_2357_WAVE18_PROOF["execution_budget"]),
        (2357, 20): (2, RELAY_2357_WAVE19_PROOF["execution_budget"]),
        (2357, 21): (2, RELAY_2357_WAVE20_PROOF["execution_budget"]),
        (2357, 22): (2, RELAY_2357_WAVE21_PROOF["execution_budget"]),
        (2357, 23): (2, RELAY_2357_WAVE22_PROOF["execution_budget"]),
        (2357, 24): (2, RELAY_2357_WAVE23_PROOF["execution_budget"]),
        (2357, 25): (2, RELAY_2357_WAVE24_PROOF["execution_budget"]),
        (2357, 26): (2, RELAY_2357_WAVE25_PROOF["execution_budget"]),
        (2357, 27): (2, RELAY_2357_WAVE26_PROOF["execution_budget"]),
        (2357, 28): (2, RELAY_2357_WAVE27_PROOF["execution_budget"]),
        (2357, 29): (2, RELAY_2357_WAVE28_PROOF["execution_budget"]),
        (2357, 30): (2, RELAY_2357_WAVE29_PROOF["execution_budget"]),
        (2357, 31): (2, RELAY_2357_WAVE30_PROOF["execution_budget"]),
        (6101, 4): (1, RELAY_6101_WAVE03_PROOF["execution_budget"]),
        (6101, 6): (1, RELAY_6101_WAVE05_PROOF["execution_budget"]),
        (6101, 8): (1, RELAY_6101_WAVE07_PROOF["execution_budget"]),
        (6101, 9): (2, CONTINUATION_6101_R5_BUDGET),
        (6101, 10): (1, RELAY_6101_WAVE09_PROOF["execution_budget"]),
        (6101, 11): (2, RELAY_6101_WAVE10_PROOF["execution_budget"]),
        (6101, 12): (2, RELAY_6101_WAVE11_PROOF["execution_budget"]),
        (6101, 13): (1, RELAY_6101_WAVE12_PROOF["execution_budget"]),
        (6101, 14): (2, RELAY_6101_WAVE13_PROOF["execution_budget"]),
        (6101, 15): (2, RELAY_6101_WAVE14_PROOF["execution_budget"]),
        (6101, 16): (2, RELAY_6101_WAVE15_PROOF["execution_budget"]),
        (6101, 17): (2, RELAY_6101_WAVE16_PROOF["execution_budget"]),
        (6101, 18): (2, RELAY_6101_WAVE17_PROOF["execution_budget"]),
        (6101, 19): (1, RELAY_6101_WAVE18_PROOF["execution_budget"]),
        (6101, 20): (2, RELAY_6101_WAVE19_PROOF["execution_budget"]),
        (6101, 21): (2, RELAY_6101_WAVE20_PROOF["execution_budget"]),
        (6101, 22): (2, RELAY_6101_WAVE21_PROOF["execution_budget"]),
        (6101, 23): (2, RELAY_6101_WAVE22_PROOF["execution_budget"]),
        (6101, 24): (2, RELAY_6101_WAVE23_PROOF["execution_budget"]),
        (6101, 25): (2, RELAY_6101_WAVE24_PROOF["execution_budget"]),
        (6101, 26): (2, RELAY_6101_WAVE25_PROOF["execution_budget"]),
        (6101, 27): (2, RELAY_6101_WAVE26_PROOF["execution_budget"]),
        (6101, 28): (2, RELAY_6101_WAVE27_PROOF["execution_budget"]),
        (887, 6): (1, RELAY_887_WAVE05_PROOF["execution_budget"]),
        (887, 7): (2, RELAY_887_WAVE06_PROOF["execution_budget"]),
        (887, 8): (1, RELAY_887_WAVE07_PROOF["execution_budget"]),
        (887, 9): (2, RELAY_887_WAVE08_PROOF["execution_budget"]),
        (887, 10): (1, RELAY_887_WAVE09_PROOF["execution_budget"]),
        (887, 11): (2, CONTINUATION_887_R20_BUDGET),
        (887, 12): (2, CONTINUATION_887_R33_BUDGET),
        (887, 13): (2, RELAY_887_WAVE12_PROOF["execution_budget"]),
        (887, 14): (2, RELAY_887_WAVE13_PROOF["execution_budget"]),
        (887, 15): (2, RELAY_887_WAVE14_PROOF["execution_budget"]),
        (157, 8): (1, RELAY_157_WAVE07_PROOF["execution_budget"]),
        (157, 9): (2, RELAY_157_WAVE08_PROOF["execution_budget"]),
        (157, 10): (2, RELAY_157_WAVE09_PROOF["execution_budget"]),
        (157, 11): (2, RELAY_157_WAVE10_PROOF["execution_budget"]),
        (157, 12): (1, RELAY_157_WAVE11_PROOF["execution_budget"]),
        (157, 13): (2, RELAY_157_WAVE12_PROOF["execution_budget"]),
        (157, 14): (2, RELAY_157_WAVE13_PROOF["execution_budget"]),
        (157, 15): (2, RELAY_157_WAVE14_PROOF["execution_budget"]),
        (157, 16): (2, RELAY_157_WAVE15_PROOF["execution_budget"]),
        (157, 17): (2, RELAY_157_WAVE16_PROOF["execution_budget"]),
        (157, 18): (2, RELAY_157_WAVE17_PROOF["execution_budget"]),
        (157, 19): (2, RELAY_157_WAVE18_PROOF["execution_budget"]),
        (157, 20): (2, RELAY_157_WAVE19_PROOF["execution_budget"]),
        (157, 21): (2, RELAY_157_WAVE20_PROOF["execution_budget"]),
        (157, 22): (2, RELAY_157_WAVE21_PROOF["execution_budget"]),
        (157, 23): (2, RELAY_157_WAVE22_PROOF["execution_budget"]),
        (157, 24): (2, RELAY_157_WAVE23_PROOF["execution_budget"]),
        (157, 25): (2, RELAY_157_WAVE24_PROOF["execution_budget"]),
        (157, 26): (2, RELAY_157_WAVE25_PROOF["execution_budget"]),
        (157, 27): (2, RELAY_157_WAVE26_PROOF["execution_budget"]),
        (157, 28): (2, RELAY_157_WAVE27_PROOF["execution_budget"]),
        (157, 29): (2, RELAY_157_WAVE28_PROOF["execution_budget"]),
        (157, 30): (2, RELAY_157_WAVE29_PROOF["execution_budget"]),
        (157, 31): (2, RELAY_157_WAVE30_PROOF["execution_budget"]),
        (12203, 8): (2, RELAY_12203_WAVE07_PROOF["execution_budget"]),
        (12203, 9): (2, RELAY_12203_WAVE08_PROOF["execution_budget"]),
        (12203, 10): (2, RELAY_12203_WAVE09_PROOF["execution_budget"]),
        (12203, 11): (1, RELAY_12203_WAVE10_PROOF["execution_budget"]),
        (12203, 12): (2, RELAY_12203_WAVE11_PROOF["execution_budget"]),
        (12203, 13): (2, RELAY_12203_WAVE12_PROOF["execution_budget"]),
        (12203, 14): (1, RELAY_12203_WAVE13_PROOF["execution_budget"]),
        (12203, 15): (2, RELAY_12203_WAVE14_PROOF["execution_budget"]),
        (12203, 16): (1, RELAY_12203_WAVE15_PROOF["execution_budget"]),
        (12203, 17): (2, RELAY_12203_WAVE16_PROOF["execution_budget"]),
        (12203, 18): (1, RELAY_12203_WAVE17_PROOF["execution_budget"]),
        (12203, 19): (2, RELAY_12203_WAVE18_PROOF["execution_budget"]),
        (12203, 20): (2, RELAY_12203_WAVE19_PROOF["execution_budget"]),
        (12203, 21): (2, RELAY_12203_WAVE20_PROOF["execution_budget"]),
        (12203, 22): (2, RELAY_12203_WAVE21_PROOF["execution_budget"]),
        (12203, 23): (2, RELAY_12203_WAVE22_PROOF["execution_budget"]),
        (12203, 24): (2, RELAY_12203_WAVE23_PROOF["execution_budget"]),
        (12203, 25): (2, RELAY_12203_WAVE24_PROOF["execution_budget"]),
        (12203, 26): (2, RELAY_12203_WAVE25_PROOF["execution_budget"]),
        (12203, 27): (2, RELAY_12203_WAVE26_PROOF["execution_budget"]),
        (12203, 28): (2, RELAY_12203_WAVE27_PROOF["execution_budget"]),
        (12203, 29): (2, RELAY_12203_WAVE28_PROOF["execution_budget"]),
        (12203, 30): (2, RELAY_12203_WAVE29_PROOF["execution_budget"]),
        (12203, 31): (2, RELAY_12203_WAVE30_PROOF["execution_budget"]),
        (12203, 32): (2, RELAY_12203_WAVE31_PROOF["execution_budget"]),
        (12203, 33): (2, RELAY_12203_WAVE32_PROOF["execution_budget"]),
        (12203, 34): (2, RELAY_12203_WAVE33_PROOF["execution_budget"]),
        (12203, 35): (2, RELAY_12203_WAVE34_PROOF["execution_budget"]),
        (12203, 36): (2, RELAY_12203_WAVE35_PROOF["execution_budget"]),
        (12203, 37): (2, RELAY_12203_WAVE36_PROOF["execution_budget"]),
    }.get((seed, wave))
    if checked_budgets is not None:
        old_physical, old_execution = checked_budgets
        old_logical = 3 if (seed, wave) == (887, 12) else 1
        original_budget = {"maximum_logical_requests": old_logical, "maximum_physical_attempts": old_physical}
        if (previous_proof.get("recovery_budget") != original_budget
                or previous_record.get("budget") != old_execution or signed.get("budget") != wave_proof["execution_budget"]
                or profile.get("sse_inactivity_seconds") != 900):
            return None
    observation = {(6101, 3): RELAY_6101_WAVE03_OBSERVATION, (6101, 4): RELAY_6101_WAVE04_OBSERVATION,
                   (6101, 5): RELAY_6101_WAVE05_OBSERVATION, (6101, 6): RELAY_6101_WAVE06_OBSERVATION,
                   (6101, 8): RELAY_6101_WAVE08_OBSERVATION, (6101, 9): RELAY_6101_WAVE09_OBSERVATION,
                   (6101, 10): RELAY_6101_WAVE10_OBSERVATION,
                   (6101, 11): RELAY_6101_WAVE11_OBSERVATION,
                   (6101, 12): RELAY_6101_WAVE12_OBSERVATION,
                   (6101, 13): RELAY_6101_WAVE13_OBSERVATION,
                   (6101, 14): RELAY_6101_WAVE14_OBSERVATION,
                   (6101, 15): RELAY_6101_WAVE15_OBSERVATION,
                   (6101, 16): RELAY_6101_WAVE16_OBSERVATION,
                   (6101, 17): RELAY_6101_WAVE17_OBSERVATION,
                   (6101, 18): RELAY_6101_WAVE18_OBSERVATION,
                   (6101, 19): RELAY_6101_WAVE19_OBSERVATION,
                   (6101, 20): RELAY_6101_WAVE20_OBSERVATION,
                   (6101, 21): RELAY_6101_WAVE21_OBSERVATION,
                   (6101, 22): RELAY_6101_WAVE22_OBSERVATION,
                   (6101, 23): RELAY_6101_WAVE23_OBSERVATION,
                   (6101, 24): RELAY_6101_WAVE24_OBSERVATION,
                   (6101, 25): RELAY_6101_WAVE25_OBSERVATION,
                   (6101, 26): RELAY_6101_WAVE26_OBSERVATION,
                   (6101, 27): RELAY_6101_WAVE27_OBSERVATION,
                   (6101, 28): RELAY_6101_WAVE28_OBSERVATION,
                   (12203, 4): RELAY_12203_WAVE04_OBSERVATION, (12203, 8): RELAY_12203_WAVE08_OBSERVATION,
                   (12203, 9): RELAY_12203_WAVE09_OBSERVATION, (12203, 10): RELAY_12203_WAVE10_OBSERVATION,
                   (12203, 11): RELAY_12203_WAVE11_OBSERVATION, (12203, 12): RELAY_12203_WAVE12_OBSERVATION,
                   (12203, 13): RELAY_12203_WAVE13_OBSERVATION, (12203, 14): RELAY_12203_WAVE14_OBSERVATION,
                   (12203, 15): RELAY_12203_WAVE15_OBSERVATION, (12203, 16): RELAY_12203_WAVE16_OBSERVATION,
                   (12203, 17): RELAY_12203_WAVE17_OBSERVATION, (12203, 18): RELAY_12203_WAVE18_OBSERVATION,
                   (12203, 19): RELAY_12203_WAVE19_OBSERVATION,
                   (12203, 20): RELAY_12203_WAVE20_OBSERVATION,
                   (12203, 21): RELAY_12203_WAVE21_OBSERVATION,
                   (12203, 22): RELAY_12203_WAVE22_OBSERVATION,
                   (12203, 23): RELAY_12203_WAVE23_OBSERVATION,
                   (12203, 24): RELAY_12203_WAVE24_OBSERVATION,
                   (12203, 25): RELAY_12203_WAVE25_OBSERVATION,
                   (12203, 26): RELAY_12203_WAVE26_OBSERVATION,
                   (12203, 27): RELAY_12203_WAVE27_OBSERVATION,
                   (12203, 28): RELAY_12203_WAVE28_OBSERVATION,
                   (12203, 29): RELAY_12203_WAVE29_OBSERVATION,
                   (12203, 30): RELAY_12203_WAVE30_OBSERVATION,
                   (12203, 31): RELAY_12203_WAVE31_OBSERVATION,
                   (12203, 32): RELAY_12203_WAVE32_OBSERVATION,
                   (12203, 33): RELAY_12203_WAVE33_OBSERVATION,
                   (12203, 34): RELAY_12203_WAVE34_OBSERVATION,
                   (12203, 35): RELAY_12203_WAVE35_OBSERVATION,
                   (12203, 36): RELAY_12203_WAVE36_OBSERVATION,
                   (12203, 37): RELAY_12203_WAVE37_OBSERVATION,
                   (157, 5): RELAY_157_WAVE05_OBSERVATION, (12203, 5): RELAY_12203_WAVE05_OBSERVATION,
                   (157, 6): RELAY_157_WAVE06_OBSERVATION, (12203, 6): RELAY_12203_WAVE06_OBSERVATION,
                   (157, 8): RELAY_157_WAVE08_OBSERVATION, (157, 9): RELAY_157_WAVE09_OBSERVATION,
                   (157, 10): RELAY_157_WAVE10_OBSERVATION, (157, 11): RELAY_157_WAVE11_OBSERVATION,
                   (157, 12): RELAY_157_WAVE12_OBSERVATION,
                   (157, 13): RELAY_157_WAVE13_OBSERVATION,
                   (157, 14): RELAY_157_WAVE14_OBSERVATION,
                   (157, 15): RELAY_157_WAVE15_OBSERVATION,
                   (157, 16): RELAY_157_WAVE16_OBSERVATION,
                   (157, 17): RELAY_157_WAVE17_OBSERVATION,
                   (157, 18): RELAY_157_WAVE18_OBSERVATION,
                   (157, 19): RELAY_157_WAVE19_OBSERVATION,
                   (157, 20): RELAY_157_WAVE20_OBSERVATION,
                   (157, 21): RELAY_157_WAVE21_OBSERVATION,
                   (157, 22): RELAY_157_WAVE22_OBSERVATION,
                   (157, 23): RELAY_157_WAVE23_OBSERVATION,
                   (157, 24): RELAY_157_WAVE24_OBSERVATION,
                   (157, 25): RELAY_157_WAVE25_OBSERVATION,
                   (157, 26): RELAY_157_WAVE26_OBSERVATION,
                   (157, 27): RELAY_157_WAVE27_OBSERVATION,
                   (157, 28): RELAY_157_WAVE28_OBSERVATION,
                   (157, 29): RELAY_157_WAVE29_OBSERVATION,
                   (157, 30): RELAY_157_WAVE30_OBSERVATION,
                   (157, 31): RELAY_157_WAVE31_OBSERVATION,
                   (887, 3): RELAY_887_WAVE03_OBSERVATION, (887, 4): RELAY_887_WAVE04_OBSERVATION,
                   (887, 6): RELAY_887_WAVE06_OBSERVATION, (887, 7): RELAY_887_WAVE07_OBSERVATION,
                   (887, 8): RELAY_887_WAVE08_OBSERVATION, (887, 9): RELAY_887_WAVE09_OBSERVATION,
                   (887, 10): RELAY_887_WAVE10_OBSERVATION,
                   (887, 11): RELAY_887_WAVE11_OBSERVATION,
                   (887, 12): RELAY_887_WAVE12_OBSERVATION,
                   (887, 13): RELAY_887_WAVE13_OBSERVATION,
                   (887, 14): RELAY_887_WAVE14_OBSERVATION,
                   (887, 15): RELAY_887_WAVE15_OBSERVATION,
                   (2357, 3): RELAY_2357_WAVE03_OBSERVATION, (2357, 4): RELAY_2357_WAVE04_OBSERVATION,
                   (2357, 5): RELAY_2357_WAVE05_OBSERVATION, (2357, 6): RELAY_2357_WAVE06_OBSERVATION,
                   (2357, 7): RELAY_2357_WAVE07_OBSERVATION, (2357, 8): RELAY_2357_WAVE08_OBSERVATION,
                   (2357, 9): RELAY_2357_WAVE09_OBSERVATION, (2357, 10): RELAY_2357_WAVE10_OBSERVATION,
                   (2357, 11): RELAY_2357_WAVE11_OBSERVATION, (2357, 12): RELAY_2357_WAVE12_OBSERVATION,
                   (2357, 13): RELAY_2357_WAVE13_OBSERVATION,
                   (2357, 14): RELAY_2357_WAVE14_OBSERVATION,
                   (2357, 15): RELAY_2357_WAVE15_OBSERVATION,
                   (2357, 16): RELAY_2357_WAVE16_OBSERVATION,
                   (2357, 17): RELAY_2357_WAVE17_OBSERVATION,
                   (2357, 18): RELAY_2357_WAVE18_OBSERVATION,
                   (2357, 19): RELAY_2357_WAVE19_OBSERVATION,
                   (2357, 20): RELAY_2357_WAVE20_OBSERVATION,
                   (2357, 21): RELAY_2357_WAVE21_OBSERVATION,
                   (2357, 22): RELAY_2357_WAVE22_OBSERVATION,
                   (2357, 23): RELAY_2357_WAVE23_OBSERVATION,
                   (2357, 24): RELAY_2357_WAVE24_OBSERVATION,
                   (2357, 25): RELAY_2357_WAVE25_OBSERVATION,
                   (2357, 26): RELAY_2357_WAVE26_OBSERVATION,
                   (2357, 27): RELAY_2357_WAVE27_OBSERVATION,
                   (2357, 28): RELAY_2357_WAVE28_OBSERVATION,
                   (2357, 29): RELAY_2357_WAVE29_OBSERVATION,
                   (2357, 30): RELAY_2357_WAVE30_OBSERVATION,
                   (2357, 31): RELAY_2357_WAVE31_OBSERVATION}.get((seed, wave))
    if observation is not None:
        sources = signed.get("source_sha256") or {}
        extended_sources = {(157, 11): RELAY_157_WAVE11_SOURCES, (2357, 10): RELAY_2357_WAVE10_SOURCES,
                            (887, 8): RELAY_887_WAVE08_SOURCES, (887, 9): RELAY_887_WAVE09_SOURCES,
                            (887, 10): RELAY_887_WAVE10_SOURCES,
                            (887, 11): RELAY_887_WAVE11_SOURCES,
                            (887, 12): RELAY_887_WAVE12_SOURCES,
                            (887, 13): RELAY_887_WAVE13_SOURCES,
                            (887, 14): RELAY_887_WAVE14_SOURCES,
                            (887, 15): RELAY_887_WAVE15_SOURCES,
                            (157, 12): RELAY_157_WAVE12_SOURCES,
                            (157, 13): RELAY_157_WAVE13_SOURCES,
                            (157, 14): RELAY_157_WAVE14_SOURCES,
                            (157, 15): RELAY_157_WAVE15_SOURCES,
                            (157, 16): RELAY_157_WAVE16_SOURCES,
                            (157, 17): RELAY_157_WAVE17_SOURCES,
                            (157, 18): RELAY_157_WAVE18_SOURCES,
                            (157, 19): RELAY_157_WAVE19_SOURCES,
                            (157, 20): RELAY_157_WAVE20_SOURCES,
                            (157, 21): RELAY_157_WAVE21_SOURCES,
                            (157, 22): RELAY_157_WAVE22_SOURCES,
                            (157, 23): RELAY_157_WAVE23_SOURCES,
                            (157, 24): RELAY_157_WAVE24_SOURCES,
                            (157, 25): RELAY_157_WAVE25_SOURCES,
                            (157, 26): RELAY_157_WAVE26_SOURCES,
                            (157, 27): RELAY_157_WAVE27_SOURCES,
                            (157, 28): RELAY_157_WAVE28_SOURCES,
                            (157, 29): RELAY_157_WAVE29_SOURCES,
                            (157, 30): RELAY_157_WAVE30_SOURCES,
                            (157, 31): RELAY_157_WAVE31_SOURCES, (12203, 17): RELAY_12203_WAVE17_SOURCES,
                            (6101, 9): RELAY_6101_WAVE09_SOURCES, (6101, 10): RELAY_6101_WAVE10_SOURCES,
                            (6101, 11): RELAY_6101_WAVE11_SOURCES,
                            (6101, 12): RELAY_6101_WAVE12_SOURCES,
                            (6101, 13): RELAY_6101_WAVE13_SOURCES,
                            (6101, 14): RELAY_6101_WAVE14_SOURCES,
                            (6101, 15): RELAY_6101_WAVE15_SOURCES,
                            (6101, 16): RELAY_6101_WAVE16_SOURCES,
                            (6101, 17): RELAY_6101_WAVE17_SOURCES,
                            (6101, 18): RELAY_6101_WAVE18_SOURCES,
                            (6101, 19): RELAY_6101_WAVE19_SOURCES,
                            (6101, 20): RELAY_6101_WAVE20_SOURCES,
                            (6101, 21): RELAY_6101_WAVE21_SOURCES,
                            (6101, 22): RELAY_6101_WAVE22_SOURCES,
                            (6101, 23): RELAY_6101_WAVE23_SOURCES,
                            (6101, 24): RELAY_6101_WAVE24_SOURCES,
                            (6101, 25): RELAY_6101_WAVE25_SOURCES,
                            (6101, 26): RELAY_6101_WAVE26_SOURCES,
                            (6101, 27): RELAY_6101_WAVE27_SOURCES,
                            (6101, 28): RELAY_6101_WAVE28_SOURCES,
                            (2357, 11): RELAY_2357_WAVE11_SOURCES,
                            (2357, 12): RELAY_2357_WAVE12_SOURCES,
                            (2357, 13): RELAY_2357_WAVE13_SOURCES,
                            (2357, 14): RELAY_2357_WAVE14_SOURCES,
                            (2357, 15): RELAY_2357_WAVE15_SOURCES,
                            (2357, 16): RELAY_2357_WAVE16_SOURCES,
                            (2357, 17): RELAY_2357_WAVE17_SOURCES,
                            (2357, 18): RELAY_2357_WAVE18_SOURCES,
                            (2357, 19): RELAY_2357_WAVE19_SOURCES,
                            (2357, 20): RELAY_2357_WAVE20_SOURCES,
                            (2357, 21): RELAY_2357_WAVE21_SOURCES,
                            (2357, 22): RELAY_2357_WAVE22_SOURCES,
                            (2357, 23): RELAY_2357_WAVE23_SOURCES,
                            (2357, 24): RELAY_2357_WAVE24_SOURCES,
                            (2357, 25): RELAY_2357_WAVE25_SOURCES,
                            (2357, 26): RELAY_2357_WAVE26_SOURCES,
                            (2357, 27): RELAY_2357_WAVE27_SOURCES,
                            (2357, 28): RELAY_2357_WAVE28_SOURCES,
                            (2357, 29): RELAY_2357_WAVE29_SOURCES,
                            (2357, 30): RELAY_2357_WAVE30_SOURCES,
                            (2357, 31): RELAY_2357_WAVE31_SOURCES,
                            (12203, 18): RELAY_12203_WAVE18_SOURCES,
                            (12203, 19): RELAY_12203_WAVE19_SOURCES,
                            (12203, 20): RELAY_12203_WAVE20_SOURCES,
                            (12203, 21): RELAY_12203_WAVE21_SOURCES,
                            (12203, 22): RELAY_12203_WAVE22_SOURCES,
                            (12203, 23): RELAY_12203_WAVE23_SOURCES,
                            (12203, 24): RELAY_12203_WAVE24_SOURCES,
                            (12203, 25): RELAY_12203_WAVE25_SOURCES,
                            (12203, 26): RELAY_12203_WAVE26_SOURCES,
                            (12203, 27): RELAY_12203_WAVE27_SOURCES,
                            (12203, 28): RELAY_12203_WAVE28_SOURCES,
                            (12203, 29): RELAY_12203_WAVE29_SOURCES,
                            (12203, 30): RELAY_12203_WAVE30_SOURCES,
                            (12203, 31): RELAY_12203_WAVE31_SOURCES,
                            (12203, 32): RELAY_12203_WAVE32_SOURCES,
                            (12203, 33): RELAY_12203_WAVE33_SOURCES,
                            (12203, 34): RELAY_12203_WAVE34_SOURCES,
                            (12203, 35): RELAY_12203_WAVE35_SOURCES,
                            (12203, 36): RELAY_12203_WAVE36_SOURCES,
                            (12203, 37): RELAY_12203_WAVE37_SOURCES}.get((seed, wave))
        same_response_guard = (fitted is not None or extended_sources is not None
                               or (seed, wave) in ((2357, 5), (2357, 6), (2357, 7), (2357, 9), (157, 8),
                                                    (887, 6), (887, 7), (6101, 8), (157, 10))
                               or (seed == 12203 and 10 <= wave <= 16))
        if extended_sources is not None:
            required_sources = extended_sources
        elif (seed, wave) == (12203, 16):
            required_sources = RELAY_12203_WAVE16_SOURCES
        elif (seed, wave) == (157, 10):
            required_sources = RELAY_157_WAVE10_SOURCES
        elif (seed, wave) == (12203, 15):
            required_sources = RELAY_12203_WAVE15_SOURCES
        elif (seed, wave) == (12203, 14):
            required_sources = RELAY_12203_WAVE14_SOURCES
        elif (seed, wave) == (12203, 13):
            required_sources = RELAY_12203_WAVE13_SOURCES
        elif (seed, wave) in ((2357, 7), (2357, 8), (2357, 9), (12203, 12), (887, 6), (887, 7), (6101, 8), (157, 9)):
            required_sources = RELAY_12203_WAVE12_SOURCES
        else:
            required_sources = RELAY_2357_WAVE05_SOURCES if same_response_guard else RELAY_RETRY_QUEUE_SOURCES
        if (proof.get("previous_transport_observation") != observation
                or any(sources.get(name) != sha for name, sha in required_sources.items())
                or same_response_guard and profile.get("request_timeouts", {}).get("total_seconds") != 3600):
            return None
    turn = directory / f"round_{start:02d}/turn_00"
    single_archive = (seed, wave) in ((2357, 6), (2357, 7), (12203, 10), (12203, 11), (12203, 14), (12203, 15),
                                      (12203, 16), (12203, 17), (12203, 18), (157, 8), (157, 9), (157, 11), (157, 12),
                                      (887, 6), (887, 7), (887, 8), (887, 9), (887, 10), (887, 11),
                                      (6101, 8), (6101, 9), (6101, 12), (6101, 13), (6101, 18), (6101, 19),
                                      (2357, 10), (2357, 11))
    if single_archive and {path.name for path in turn.glob("attempt_*.json")} != {
        "attempt_00.started.json", "attempt_00.terminal.json",
    }:
        return None
    if (seed, wave) in ((6101, 9), (6101, 12), (6101, 13), (887, 10), (887, 11)) and (
        set(turn.parent.glob("turn_*/attempt_*.started.json")) != {turn / "attempt_00.started.json"}
        or any((turn.parent / name).exists() for name in ("batch_freeze.json", "selection.json"))
    ):
        return None
    if (seed, wave) in ((887, 12), (887, 13), (887, 14), (887, 15), (12203, 13), (12203, 19), (12203, 20), (12203, 21), (12203, 22), (12203, 23),
                        (12203, 24), (12203, 25), (12203, 26), (12203, 27), (12203, 28), (12203, 29), (12203, 30), (12203, 31), (12203, 32), (12203, 33), (12203, 34), (12203, 35), (12203, 36), (12203, 37),
                        (2357, 8), (2357, 9), (2357, 12), (2357, 13), (2357, 14), (2357, 15), (2357, 16), (2357, 17), (2357, 18), (2357, 19), (2357, 20), (2357, 21), (2357, 22), (2357, 23), (2357, 24), (2357, 25), (2357, 26), (2357, 27), (2357, 28), (2357, 29), (2357, 30), (2357, 31),
                        (157, 10), (157, 13), (157, 14), (157, 15), (157, 16), (157, 17), (157, 18), (157, 19), (157, 20), (157, 21), (157, 22), (157, 23), (157, 24), (157, 25), (157, 26), (157, 27), (157, 28), (157, 29), (157, 30), (157, 31),
                        (6101, 10), (6101, 11), (6101, 14), (6101, 15), (6101, 16), (6101, 17), (6101, 20), (6101, 21), (6101, 22), (6101, 23), (6101, 24), (6101, 25), (6101, 26), (6101, 27), (6101, 28)) and (
        set(turn.parent.glob("turn_*/attempt_*.started.json")) != {turn / f"attempt_{index:02d}.started.json" for index in (0, 1)}
        or any(turn.glob("*.reconciliation.json")) or any(turn.glob("*.recovery.terminal.json"))
        or (((seed, wave) == (887, 12) or post_r34_retry) and any(
            (turn.parent / name).exists() for name in ("batch_freeze.json", "selection.json", "memory.json")))
    ):
        return None
    multiple_previous = {
        (157, 31): (RELAY_157_WAVE30_ATTEMPTS, RELAY_157_WAVE30_TERMINALS, RELAY_157_WAVE30_REPLAY_FILES),
        (2357, 31): (RELAY_2357_WAVE30_ATTEMPTS, RELAY_2357_WAVE30_TERMINALS, RELAY_2357_WAVE30_REPLAY_FILES),
        (12203, 37): (RELAY_12203_WAVE36_ATTEMPTS, RELAY_12203_WAVE36_TERMINALS, RELAY_12203_WAVE36_REPLAY_FILES),
        (12203, 36): (RELAY_12203_WAVE35_ATTEMPTS, RELAY_12203_WAVE35_TERMINALS, RELAY_12203_WAVE35_REPLAY_FILES),
        (2357, 30): (RELAY_2357_WAVE29_ATTEMPTS, RELAY_2357_WAVE29_TERMINALS, RELAY_2357_WAVE29_REPLAY_FILES),
        (2357, 29): (RELAY_2357_WAVE28_ATTEMPTS, RELAY_2357_WAVE28_TERMINALS, RELAY_2357_WAVE28_REPLAY_FILES),
        (887, 15): (RELAY_887_WAVE14_ATTEMPTS, RELAY_887_WAVE14_TERMINALS, RELAY_887_WAVE14_REPLAY_FILES),
        (887, 14): (RELAY_887_WAVE13_ATTEMPTS, RELAY_887_WAVE13_TERMINALS, RELAY_887_WAVE13_REPLAY_FILES),
        (157, 30): (RELAY_157_WAVE29_ATTEMPTS, RELAY_157_WAVE29_TERMINALS, RELAY_157_WAVE29_REPLAY_FILES),
        (157, 29): (RELAY_157_WAVE28_ATTEMPTS, RELAY_157_WAVE28_TERMINALS, RELAY_157_WAVE28_REPLAY_FILES),
        (12203, 35): (RELAY_12203_WAVE34_ATTEMPTS, RELAY_12203_WAVE34_TERMINALS, RELAY_12203_WAVE34_REPLAY_FILES),
        (6101, 28): (RELAY_6101_WAVE27_ATTEMPTS, RELAY_6101_WAVE27_TERMINALS, RELAY_6101_WAVE27_REPLAY_FILES),
        (2357, 28): (RELAY_2357_WAVE27_ATTEMPTS, RELAY_2357_WAVE27_TERMINALS, RELAY_2357_WAVE27_REPLAY_FILES),
        (887, 13): (RELAY_887_WAVE13_PARENT_ATTEMPTS, RELAY_887_WAVE13_PARENT_TERMINALS, RELAY_887_WAVE13_PARENT_REPLAY_FILES),
        (157, 28): (RELAY_157_WAVE27_ATTEMPTS, RELAY_157_WAVE27_TERMINALS, RELAY_157_WAVE27_REPLAY_FILES),
        (887, 12): (RELAY_887_WAVE12_PREVIOUS_ATTEMPTS, RELAY_887_WAVE12_TERMINALS, RELAY_887_WAVE12_REPLAY_FILES),
        (12203, 34): (RELAY_12203_WAVE33_ATTEMPTS, RELAY_12203_WAVE33_TERMINALS, RELAY_12203_WAVE33_REPLAY_FILES),
        (6101, 27): (RELAY_6101_WAVE26_ATTEMPTS, RELAY_6101_WAVE26_TERMINALS, RELAY_6101_WAVE26_REPLAY_FILES),
        (2357, 27): (RELAY_2357_WAVE26_ATTEMPTS, RELAY_2357_WAVE26_TERMINALS, RELAY_2357_WAVE26_REPLAY_FILES),
        (157, 27): (RELAY_157_WAVE26_ATTEMPTS, RELAY_157_WAVE26_TERMINALS, RELAY_157_WAVE26_REPLAY_FILES),
        (12203, 33): (RELAY_12203_WAVE32_ATTEMPTS, RELAY_12203_WAVE32_TERMINALS, RELAY_12203_WAVE32_REPLAY_FILES),
        (157, 26): (RELAY_157_WAVE25_ATTEMPTS, RELAY_157_WAVE25_TERMINALS, RELAY_157_WAVE25_REPLAY_FILES),
        (6101, 26): (RELAY_6101_WAVE25_ATTEMPTS, RELAY_6101_WAVE25_TERMINALS, RELAY_6101_WAVE25_REPLAY_FILES),
        (2357, 26): (RELAY_2357_WAVE25_ATTEMPTS, RELAY_2357_WAVE25_TERMINALS, RELAY_2357_WAVE25_REPLAY_FILES),
        (12203, 32): (RELAY_12203_WAVE31_ATTEMPTS, RELAY_12203_WAVE31_TERMINALS, RELAY_12203_WAVE31_REPLAY_FILES),
        (6101, 25): (RELAY_6101_WAVE24_ATTEMPTS, RELAY_6101_WAVE24_TERMINALS, RELAY_6101_WAVE24_REPLAY_FILES),
        (2357, 25): (RELAY_2357_WAVE24_ATTEMPTS, RELAY_2357_WAVE24_TERMINALS, RELAY_2357_WAVE24_REPLAY_FILES),
        (157, 25): (RELAY_157_WAVE24_ATTEMPTS, RELAY_157_WAVE24_TERMINALS, RELAY_157_WAVE24_REPLAY_FILES),
        (12203, 31): (RELAY_12203_WAVE30_ATTEMPTS, RELAY_12203_WAVE30_TERMINALS, RELAY_12203_WAVE30_REPLAY_FILES),
        (6101, 24): (RELAY_6101_WAVE23_ATTEMPTS, RELAY_6101_WAVE23_TERMINALS, RELAY_6101_WAVE23_REPLAY_FILES),
        (2357, 24): (RELAY_2357_WAVE23_ATTEMPTS, RELAY_2357_WAVE23_TERMINALS, RELAY_2357_WAVE23_REPLAY_FILES),
        (157, 24): (RELAY_157_WAVE23_ATTEMPTS, RELAY_157_WAVE23_TERMINALS, RELAY_157_WAVE23_REPLAY_FILES),
        (12203, 30): (RELAY_12203_WAVE29_ATTEMPTS, RELAY_12203_WAVE29_TERMINALS, RELAY_12203_WAVE29_REPLAY_FILES),
        (157, 23): (RELAY_157_WAVE22_ATTEMPTS, RELAY_157_WAVE22_TERMINALS, RELAY_157_WAVE22_REPLAY_FILES),
        (6101, 23): (RELAY_6101_WAVE22_ATTEMPTS, RELAY_6101_WAVE22_TERMINALS, RELAY_6101_WAVE22_REPLAY_FILES),
        (2357, 23): (RELAY_2357_WAVE22_ATTEMPTS, RELAY_2357_WAVE22_TERMINALS, RELAY_2357_WAVE22_REPLAY_FILES),
        (12203, 29): (RELAY_12203_WAVE28_ATTEMPTS, RELAY_12203_WAVE28_TERMINALS, RELAY_12203_WAVE28_REPLAY_FILES),
        (157, 22): (RELAY_157_WAVE21_ATTEMPTS, RELAY_157_WAVE21_TERMINALS, RELAY_157_WAVE21_REPLAY_FILES),
        (6101, 22): (RELAY_6101_WAVE21_ATTEMPTS, RELAY_6101_WAVE21_TERMINALS, RELAY_6101_WAVE21_REPLAY_FILES),
        (2357, 22): (RELAY_2357_WAVE21_ATTEMPTS, RELAY_2357_WAVE21_TERMINALS, RELAY_2357_WAVE21_REPLAY_FILES),
        (12203, 28): (RELAY_12203_WAVE27_ATTEMPTS, RELAY_12203_WAVE27_TERMINALS, RELAY_12203_WAVE27_REPLAY_FILES),
        (6101, 21): (RELAY_6101_WAVE20_ATTEMPTS, RELAY_6101_WAVE20_TERMINALS, RELAY_6101_WAVE20_REPLAY_FILES),
        (2357, 21): (RELAY_2357_WAVE20_ATTEMPTS, RELAY_2357_WAVE20_TERMINALS, RELAY_2357_WAVE20_REPLAY_FILES),
        (157, 21): (RELAY_157_WAVE20_ATTEMPTS, RELAY_157_WAVE20_TERMINALS, RELAY_157_WAVE20_REPLAY_FILES),
        (12203, 27): (RELAY_12203_WAVE26_ATTEMPTS, RELAY_12203_WAVE26_TERMINALS, RELAY_12203_WAVE26_REPLAY_FILES),
        (6101, 20): (RELAY_6101_WAVE19_ATTEMPTS, RELAY_6101_WAVE19_TERMINALS, RELAY_6101_WAVE19_REPLAY_FILES),
        (2357, 20): (RELAY_2357_WAVE19_ATTEMPTS, RELAY_2357_WAVE19_TERMINALS, RELAY_2357_WAVE19_REPLAY_FILES),
        (157, 20): (RELAY_157_WAVE19_ATTEMPTS, RELAY_157_WAVE19_TERMINALS, RELAY_157_WAVE19_REPLAY_FILES),
        (12203, 26): (RELAY_12203_WAVE25_ATTEMPTS, RELAY_12203_WAVE25_TERMINALS, RELAY_12203_WAVE25_REPLAY_FILES),
        (12203, 25): (RELAY_12203_WAVE24_ATTEMPTS, RELAY_12203_WAVE24_TERMINALS, RELAY_12203_WAVE24_REPLAY_FILES),
        (2357, 19): (RELAY_2357_WAVE18_ATTEMPTS, RELAY_2357_WAVE18_TERMINALS, RELAY_2357_WAVE18_REPLAY_FILES),
        (157, 19): (RELAY_157_WAVE18_ATTEMPTS, RELAY_157_WAVE18_TERMINALS, RELAY_157_WAVE18_REPLAY_FILES),
        (157, 18): (RELAY_157_WAVE17_ATTEMPTS, RELAY_157_WAVE17_TERMINALS, RELAY_157_WAVE17_REPLAY_FILES),
        (6101, 17): (RELAY_6101_WAVE16_ATTEMPTS, RELAY_6101_WAVE16_TERMINALS, RELAY_6101_WAVE16_REPLAY_FILES),
        (12203, 24): (RELAY_12203_WAVE23_ATTEMPTS, RELAY_12203_WAVE23_TERMINALS, RELAY_12203_WAVE23_REPLAY_FILES),
        (2357, 18): (RELAY_2357_WAVE17_ATTEMPTS, RELAY_2357_WAVE17_TERMINALS, RELAY_2357_WAVE17_REPLAY_FILES),
        (157, 17): (RELAY_157_WAVE16_ATTEMPTS, RELAY_157_WAVE16_TERMINALS, RELAY_157_WAVE16_REPLAY_FILES),
        (6101, 16): (RELAY_6101_WAVE15_ATTEMPTS, RELAY_6101_WAVE15_TERMINALS, RELAY_6101_WAVE15_REPLAY_FILES),
        (12203, 23): (RELAY_12203_WAVE22_ATTEMPTS, RELAY_12203_WAVE22_TERMINALS, RELAY_12203_WAVE22_REPLAY_FILES),
        (2357, 17): (RELAY_2357_WAVE16_ATTEMPTS, RELAY_2357_WAVE16_TERMINALS, RELAY_2357_WAVE16_REPLAY_FILES),
        (12203, 22): (RELAY_12203_WAVE21_ATTEMPTS, RELAY_12203_WAVE21_TERMINALS, RELAY_12203_WAVE21_REPLAY_FILES),
        (6101, 15): (RELAY_6101_WAVE14_ATTEMPTS, RELAY_6101_WAVE14_TERMINALS, RELAY_6101_WAVE14_REPLAY_FILES),
        (2357, 16): (RELAY_2357_WAVE15_ATTEMPTS, RELAY_2357_WAVE15_TERMINALS, RELAY_2357_WAVE15_REPLAY_FILES),
        (157, 16): (RELAY_157_WAVE15_ATTEMPTS, RELAY_157_WAVE15_TERMINALS, RELAY_157_WAVE15_REPLAY_FILES),
        (157, 15): (RELAY_157_WAVE14_ATTEMPTS, RELAY_157_WAVE14_TERMINALS, RELAY_157_WAVE14_REPLAY_FILES),
        (6101, 14): (RELAY_6101_WAVE13_ATTEMPTS, RELAY_6101_WAVE13_TERMINALS, RELAY_6101_WAVE13_REPLAY_FILES),
        (12203, 21): (RELAY_12203_WAVE20_ATTEMPTS, RELAY_12203_WAVE20_TERMINALS, RELAY_12203_WAVE20_REPLAY_FILES),
        (2357, 15): (RELAY_2357_WAVE14_ATTEMPTS, RELAY_2357_WAVE14_TERMINALS, RELAY_2357_WAVE14_REPLAY_FILES),
        (157, 14): (RELAY_157_WAVE13_ATTEMPTS, RELAY_157_WAVE13_TERMINALS, RELAY_157_WAVE13_REPLAY_FILES),
        (12203, 20): (RELAY_12203_WAVE19_ATTEMPTS, RELAY_12203_WAVE19_TERMINALS, RELAY_12203_WAVE19_REPLAY_FILES),
        (2357, 14): (RELAY_2357_WAVE13_ATTEMPTS, RELAY_2357_WAVE13_TERMINALS, RELAY_2357_WAVE13_REPLAY_FILES),
        (157, 13): (RELAY_157_WAVE12_ATTEMPTS, RELAY_157_WAVE12_TERMINALS, RELAY_157_WAVE12_REPLAY_FILES),
        (12203, 19): (RELAY_12203_WAVE18_ATTEMPTS, RELAY_12203_WAVE18_TERMINALS, RELAY_12203_WAVE18_REPLAY_FILES),
        (6101, 11): (RELAY_6101_WAVE10_ATTEMPTS, RELAY_6101_WAVE10_TERMINALS, RELAY_6101_WAVE10_REPLAY_FILES),
        (2357, 13): (RELAY_2357_WAVE12_ATTEMPTS, RELAY_2357_WAVE12_TERMINALS, RELAY_2357_WAVE12_REPLAY_FILES),
        (2357, 4): (RELAY_2357_WAVE03_ATTEMPTS, RELAY_2357_WAVE03_TERMINALS, RELAY_2357_WAVE03_REPLAY_FILES),
        (2357, 8): (RELAY_2357_WAVE07_ATTEMPTS, RELAY_2357_WAVE07_TERMINALS, RELAY_2357_WAVE07_REPLAY_FILES),
        (2357, 9): (RELAY_2357_WAVE08_ATTEMPTS, RELAY_2357_WAVE08_TERMINALS, RELAY_2357_WAVE08_REPLAY_FILES),
        (2357, 12): (RELAY_2357_WAVE11_ATTEMPTS, RELAY_2357_WAVE11_TERMINALS, RELAY_2357_WAVE11_REPLAY_FILES),
        (6101, 10): (RELAY_6101_WAVE09_ATTEMPTS, RELAY_6101_WAVE09_TERMINALS, RELAY_6101_WAVE09_REPLAY_FILES),
        (157, 10): (RELAY_157_WAVE09_ATTEMPTS, RELAY_157_WAVE09_TERMINALS, RELAY_157_WAVE09_REPLAY_FILES),
        (12203, 8): (RELAY_12203_WAVE07_ATTEMPTS, RELAY_12203_WAVE07_TERMINALS, RELAY_12203_WAVE07_REPLAY_FILES),
        (12203, 9): (RELAY_12203_WAVE08_ATTEMPTS, RELAY_12203_WAVE08_TERMINALS, RELAY_12203_WAVE08_REPLAY_FILES),
        (12203, 12): (RELAY_12203_WAVE11_ATTEMPTS, RELAY_12203_WAVE11_TERMINALS, RELAY_12203_WAVE11_REPLAY_FILES),
        (12203, 13): (RELAY_12203_WAVE12_ATTEMPTS, RELAY_12203_WAVE12_TERMINALS, RELAY_12203_WAVE12_REPLAY_FILES),
    }.get((seed, wave))
    if multiple_previous is not None:
        old_attempts, terminals, replay_files = multiple_previous
        if (profile.get("sse_inactivity_seconds") != 900
                or any(checksum(turn / name) != sha for name, sha in replay_files.items())):
            return None
        attempts = _reviewed_fitted_relay_receipts(directory, start, proof, {}, old_attempts, terminals)
        if attempts is None:
            return None
    else:
        attempts = proof.get("previous_attempts")
        if type(attempts) is not list or len(attempts) != 1 or type(attempts[0]) is not dict:
            return None
        attempt = attempts[0]
        if (not isinstance(attempt.get("terminal_sha256"), str) or not SHA.fullmatch(attempt["terminal_sha256"])
                or record_digest({key: attempt.get(key) for key in prior_attempt}) != record_digest(prior_attempt)):
            return None
        if wave >= 4 or observation is not None:
            terminal = turn / "attempt_00.terminal.json"
            if (checksum(terminal) != proof["previous_terminal_file_sha256"]
                    or record_digest(read_json(terminal)) != attempt["terminal_sha256"]):
                return None
    prefix = _reviewed_relay_prefix(runs, seed, wave, previous_proof)
    latest = _relay_review(previous, previous_contract, pinned_auth, proof, attempts)
    if prefix is None or proof.get("reviewed_retry_chain") != [*prefix, latest]:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def relay_retry_07_12203_contract(runs, previous, previous_contract):
    seed = 12203
    if previous_contract != RELAY_12203_WAVE06_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-07"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 2,
                                           previous_suffix="relay-retry-06")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old = read_json(previous / "formal_contract.json")
    old_profile = old.get("profile", {})
    boundaries = profile.get("resume_boundaries", {})
    signed = read_json(run / "formal_contract.json")
    sources = signed.get("source_sha256", {})
    exact = {**RELAY_12203_WAVE07_PROOF, "previous_authorization_sha256": RELAY_12203_WAVE06_AUTH,
             "reviewed_fitted_prefix_files": RELAY_12203_WAVE06_FITTED_PREFIX,
             "previous_transport_observation": RELAY_12203_WAVE07_OBSERVATION}
    if (start != 5 or contract != RELAY_12203_WAVE07_CONTRACT
            or record_digest(read_json(run / "authorization.json")) != RELAY_12203_WAVE07_AUTH
            or signed.get("budget") != RELAY_12203_WAVE07_PROOF["execution_budget"]
            or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 4
            or (old.get("predecessor") or {}).get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
            or old.get("budget") != {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187}
            or set(boundaries) != {str(seed)} or profile != {**old_profile, "resume_boundaries": boundaries,
                                                              "replay_eligible_from_round": 5}
            or profile.get("audited_future_transport_replay") is not True
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or any(sources.get(name) != sha for name, sha in RELAY_RETRY_QUEUE_SOURCES.items())
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    directory = previous / f"seed_{seed}"
    turn = directory / "round_05/turn_00"
    if any(checksum(turn / name) != sha for name, sha in RELAY_12203_WAVE06_REPLAY_FILES.items()):
        return None
    receipts = _reviewed_fitted_relay_receipts(directory, start, proof, RELAY_12203_WAVE06_FITTED_PREFIX,
                                              RELAY_12203_WAVE06_ATTEMPTS, RELAY_12203_WAVE06_TERMINALS)
    if receipts is None or proof.get("reviewed_retry_chain") != [_relay_review(previous, previous_contract,
                                                                                RELAY_12203_WAVE06_AUTH, proof, receipts)]:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def cached_r3_157_contract(runs, seed, previous, previous_contract):
    if seed != 157 or previous_contract != CACHED_R3_157_OLD_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-157-cached-r3-01"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_157_cached_r3", 0,
                                           previous_suffix="relay-retry-06")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    cached = proof.get("cached_import") or {}
    old_turn = previous / "seed_157/round_03/turn_00"
    cached_attempt = CACHED_R3_157_REVIEW["attempt_id"]
    old_job = read_json(previous / "bridge/outbox" / f"{cached_attempt}.json")
    old_result = read_json(previous / "bridge/results" / f"{cached_attempt}.json")
    if (contract != CACHED_R3_157_CONTRACT or start != 3
            or record_digest(read_json(run / "authorization.json")) != CACHED_R3_157_AUTH
            or proof.get("previous_authorization_sha256") != CACHED_R3_157_OLD_AUTH
            or proof.get("execution_budget") != CACHED_R3_157_BUDGET
            or profile.get("replay_eligible_from_round") != 4 or profile.get("audited_future_transport_replay") is not True
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or record_digest({key: cached.get(key) for key in CACHED_R3_157_REVIEW}) != record_digest(CACHED_R3_157_REVIEW)
            or not isinstance(mirrors := cached.get("removed_paths"), list) or len(mirrors) != 79
            or record_digest(mirrors) != CACHED_R3_157_MIRRORS_SHA
            or checksum(run / "cached_response_import.json") != CACHED_R3_157_IMPORT_FILE
            or read_json(run / "cached_response_import.json") != cached
            or checksum(old_turn / "attempt_00.terminal.json") != CACHED_R3_157_REVIEW["archived_terminal_file_sha256"]
            or checksum(old_turn / "decision.json") != CACHED_R3_157_REVIEW["archived_decision_file_sha256"]
            or old_job.get("job_sha256") != CACHED_R3_157_REVIEW["job_sha256"]
            or old_result.get("result_sha256") != CACHED_R3_157_REVIEW["result_sha256"]
            or old_result.get("raw_response_sha256") != CACHED_R3_157_REVIEW["raw_response_sha256"]):
        return None
    directory = run / "seed_157"
    terminal = read_json(directory / "round_03/turn_00/attempt_00.terminal.json")
    local_attempt = terminal.get("attempt_id")
    fitted = directory / "round_04"
    transition = read_json(fitted / "transition.json")
    fit = transition.get("fit_path")
    if (any(checksum(directory / name) != sha for name, sha in CACHED_R3_157_FITTED_FILES.items())
            or not isinstance(local_attempt, str) or not IDENTITY.fullmatch(local_attempt) or local_attempt == cached_attempt
            or terminal.get("status") != "completed" or terminal.get("provider_dispatched") is not False
            or terminal.get("cached_import_sha256") != record_digest(cached)
            or terminal.get("imported_from_response_id") != CACHED_R3_157_PROVIDER_RESPONSE
            or (run / "bridge/outbox" / f"{local_attempt}.json").exists()
            or not isinstance(fit, str) or not FIT.fullmatch(fit)
            or transition.get("checkpoint_sha256") != CACHED_R3_157_CHECKPOINT
            or checksum(fitted / "source.json") != transition.get("source_sha256")
            or transition.get("fit_audit_sha256") != CACHED_R3_157_FIT_AUDIT
            or checksum(fitted / fit / "fit_audit.json") != CACHED_R3_157_FIT_AUDIT
            or transition.get("validation_metrics_sha256") != CACHED_R3_157_METRICS
            or checksum(fitted / fit / "validation_metrics.json") != CACHED_R3_157_METRICS
            or read_json(fitted / "source.json").get("checkpoint") != CACHED_R3_157_CHECKPOINT):
        return None
    return run, contract, start, profile["ipv4_readiness_windows"]


def cached_r2_6101_contract(runs, seed, previous, previous_contract):
    if seed != 6101 or previous_contract != RELAY_6101_WAVE06_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-6101-cached-r2-01"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_6101_cached_r2", 0,
                                           previous_suffix="relay-retry-06")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    cached = proof.get("cached_import") or {}
    mirrors = cached.get("removed_paths")
    transport = profile.get("transport") or {}
    review = CACHED_R2_6101_REVIEW
    old_attempt = review["attempt_id"]
    old_turn = previous / "seed_6101/round_02/turn_00"
    old_job = read_json(previous / "bridge/outbox" / f"{old_attempt}.json")
    old_result = read_json(previous / "bridge/results" / f"{old_attempt}.json")
    if (contract != CACHED_R2_6101_CONTRACT or start != 2
            or record_digest(read_json(run / "authorization.json")) != CACHED_R2_6101_AUTH
            or proof.get("previous_authorization_sha256") != RELAY_6101_WAVE06_AUTH
            or proof.get("execution_budget") != CACHED_R2_6101_BUDGET
            or profile.get("replay_eligible_from_round") != 3 or profile.get("audited_future_transport_replay") is not True
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or (transport.get("logical_replies"), transport.get("transport_attempts")) != (3, 2)
            or record_digest({key: cached.get(key) for key in review}) != record_digest(review)
            or not isinstance(mirrors, list) or record_digest(mirrors) != CACHED_R2_6101_MIRRORS_SHA
            or read_json(run / "cached_response_import.json") != cached
            or checksum(old_turn / "attempt_00.terminal.json") != review["archived_terminal_file_sha256"]
            or checksum(old_turn / "decision.json") != review["archived_decision_file_sha256"]
            or old_job.get("job_sha256") != review["job_sha256"]
            or old_result.get("result_sha256") != review["result_sha256"]
            or old_result.get("raw_response_sha256") != review["raw_response_sha256"]
            or old_result.get("provider_dispatched") is not True):
        return None
    observed = sum(isinstance(path, str) and path.endswith("/observed_id") for path in mirrors)
    candidates = sum(isinstance(path, str) and path.endswith("/candidate_id") for path in mirrors)
    if (observed, candidates, len(mirrors)) != (59, 15, 74):
        return None
    directory = run / "seed_6101"
    terminal = read_json(directory / "round_02/turn_00/attempt_00.terminal.json")
    local_attempt = terminal.get("attempt_id")
    fitted = directory / "round_03"
    transition_file = fitted / "transition.json"
    transition = read_json(transition_file)
    fit = transition.get("fit_path")
    if (checksum(transition_file) != CACHED_R2_6101_TRANSITION
            or not isinstance(local_attempt, str) or not IDENTITY.fullmatch(local_attempt) or local_attempt == old_attempt
            or terminal.get("status") != "completed" or terminal.get("provider_dispatched") is not False
            or terminal.get("usage") != {} or terminal.get("cached_import_sha256") != record_digest(cached)
            or terminal.get("imported_from_response_id") != CACHED_R2_6101_RESPONSE
            or (run / "bridge/outbox" / f"{local_attempt}.json").exists()
            or not isinstance(fit, str) or not FIT.fullmatch(fit)
            or transition.get("checkpoint_sha256") != CACHED_R2_6101_CHECKPOINT
            or checksum(fitted / "source.json") != transition.get("source_sha256")
            or read_json(fitted / "source.json").get("checkpoint") != CACHED_R2_6101_CHECKPOINT):
        return None
    receipts = {"fit_audit_sha256": fitted / fit / "fit_audit.json",
                "validation_metrics_sha256": fitted / fit / "validation_metrics.json"}
    if not all(transition.get(key) and checksum(path) == transition[key] for key, path in receipts.items()):
        return None
    report = read_json(receipts["validation_metrics_sha256"])
    imported = {"round": start, "fitted_labels": 429, "choice_count": 32, "removed_reference_fields": len(mirrors),
                "removed_observed_reference_fields": observed, "removed_candidate_reference_fields": candidates,
                "provider_dispatched": False, "checkpoint_sha256": CACHED_R2_6101_CHECKPOINT,
                "validation_nrmse": report.get("combined_normalized_rmse"),
                "future_starts_at_round": 3, "future_logical_replies": 3, "future_transport_attempts": 2}
    return run, contract, start, profile["ipv4_readiness_windows"], imported


def _cached_r4_6101_fit(directory, custody, contract):
    round_dir, fitted = (directory / name for name in ("round_04", "round_05"))
    turn = round_dir / "turn_00"
    terminal_file = turn / "attempt_00.terminal.json"
    transition_file = fitted / "transition.json"
    if (checksum(terminal_file), checksum(transition_file)) != (CACHED_R4_6101_LOCAL_TERMINAL, CACHED_R4_6101_TRANSITION):
        return None
    terminal = read_json(terminal_file)
    plan = read_json(round_dir / "selection.json")
    freeze_file = round_dir / "batch_freeze.json"
    freeze = read_json(freeze_file)
    binding = freeze.get("binding") or {}
    formal_run = binding.get("a7_formal") or {}
    batch = freeze.get("batch_ids")
    transition = read_json(transition_file)
    fit = transition.get("fit_path")
    if (not isinstance(fit, str) or not FIT.fullmatch(fit) or not isinstance(batch, list) or len(batch) != 32
            or record_digest(batch) != custody["frozen_order_sha256"]
            or record_digest(plan) != custody["normalized_plan_sha256"]
            or formal_run.get("contract_sha256") != contract
            or (binding.get("seed"), binding.get("round"), binding.get("checkpoint_sha256"))
            != (6101, 4, CACHED_R4_6101_PREVIOUS_CHECKPOINT)
            or freeze.get("label_access") is not False or freeze.get("mock") is not False
            or (freeze.get("accepted_logical_index"), freeze.get("accepted_attempt")) != (0, 0)
            or (terminal.get("status"), terminal.get("provider_dispatched"), terminal.get("usage")) != ("completed", False, {})
            or (terminal.get("cached_selection_sha256"), terminal.get("imported_from_response_id"))
            != (record_digest(custody), custody["response_id"])
            or terminal.get("response_id") != freeze.get("provider_response_id")
            or terminal.get("request_id") != freeze.get("request_id") or terminal.get("request_id") == custody["request_id"]
            or read_json(turn / "decision.json").get("state") != "legal"
            or checksum(round_dir / "selection.json") != freeze.get("selection_file_sha256")
            or checksum(round_dir / "memory.json") != freeze.get("memory_file_sha256")
            or transition.get("formal_run") != formal_run
            or transition.get("selection_freeze_sha256") != checksum(freeze_file)
            or transition.get("memory_file_sha256") != freeze.get("memory_file_sha256")
            or transition.get("checkpoint_sha256") != CACHED_R4_6101_FITTED_CHECKPOINT):
        return None
    fit_dir = fitted / fit
    files = {"fit_audit_sha256": fit_dir / "fit_audit.json", "fit_artifact_receipt_sha256": fit_dir / "fit_artifact_receipt.json",
             "validation_metrics_sha256": fit_dir / "validation_metrics.json", "feedback_sha256": fitted / "authorized_feedback.json",
             "source_sha256": fitted / "source.json"}
    if any(not transition.get(key) or checksum(path) != transition[key] for key, path in files.items()):
        return None
    grant = read_json(fitted / "fit_authorization.json")
    feedback = read_json(files["feedback_sha256"])
    audit = read_json(files["fit_audit_sha256"])
    receipt = read_json(files["fit_artifact_receipt_sha256"])
    metrics = read_json(files["validation_metrics_sha256"])
    source = read_json(files["source_sha256"])
    checkpoint = CACHED_R4_6101_FITTED_CHECKPOINT
    if (grant != {"scope": "reveal_and_fit_one_frozen_batch", "formal_run": formal_run, "device": "cuda",
                  "selection_freeze_sha256": transition["selection_freeze_sha256"]}
            or feedback.get("formal_grant_sha256") != record_digest(grant)
            or feedback.get("selection_freeze_sha256") != transition["selection_freeze_sha256"]
            or [row.get("candidate_id") for row in feedback.get("records", [])] != batch
            or (audit.get("checkpoint_sha256"), audit.get("train_rows"), source.get("checkpoint"), source.get("round"))
            != (checkpoint, 493, checkpoint, 5)
            or metrics.get("checkpoint_sha256") != checkpoint or receipt.get("best.pt") != checkpoint
            or checksum(fit_dir / "best.pt") != checkpoint):
        return None
    return {"checkpoint_sha256": checkpoint, "validation_nrmse": metrics.get("combined_normalized_rmse")}


def cached_r4_6101_contract(runs, seed, previous, previous_contract):
    if seed != 6101 or previous_contract != RELAY_6101_WAVE08_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-6101-cached-r4-01"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, CACHED_R4_6101_MODULE, 0,
                                           previous_suffix="relay-retry-08", target_labels=493)
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    cached = proof.get("cached_selection") or {}
    old = read_json(previous / "formal_contract.json")
    current = read_json(run / "formal_contract.json")
    module = f"studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/{CACHED_R4_6101_MODULE}.py"
    old_profile = old.get("profile", {})
    transport = {**old_profile.get("transport", {}), "logical_replies": 1, "transport_attempts": 1}
    inherited = {key: value for key, value in old_profile.items()
                 if key not in ("audited_future_transport_replay", "replay_eligible_from_round", "sealed_relay_retry_seconds")}
    boundaries = profile.get("resume_boundaries", {})
    mirrors = cached.get("removed_paths")
    if (contract != CACHED_R4_6101_CONTRACT or start != 4
            or record_digest(read_json(run / "authorization.json")) != CACHED_R4_6101_AUTH
            or proof.get("previous_authorization_sha256") != RELAY_6101_WAVE08_AUTH
            or current.get("budget") != CACHED_R4_6101_BUDGET or proof.get("execution_budget") != CACHED_R4_6101_BUDGET
            or proof.get("newly_authorized_boundary_physical_attempts") != 0
            or proof.get("checkpoint_sha256") != CACHED_R4_6101_PREVIOUS_CHECKPOINT
            or current.get("source_sha256") != {**old.get("source_sha256", {}), module: CACHED_R4_6101_SOURCE}
            or set(boundaries) != {str(seed)}
            or profile != {**inherited, "target_labels": 493, "resume_boundaries": boundaries, "transport": transport}
            or record_digest({key: cached.get(key) for key in CACHED_R4_6101_REVIEW}) != record_digest(CACHED_R4_6101_REVIEW)
            or cached.get("archived_boundary_files") != CACHED_R4_6101_BOUNDARY_FILES
            or not isinstance(mirrors, list) or len(mirrors) != 42 or record_digest(mirrors) != CACHED_R4_6101_MIRRORS_SHA
            or read_json(run / "cached_response_import.json") != cached):
        return None
    old_directory = previous / "seed_6101"
    archived = old_directory / "round_04"
    original_job = read_json(previous / "bridge/outbox" / f"{cached['attempt_id']}.json")
    original_result = read_json(previous / "bridge/results" / f"{cached['attempt_id']}.json")
    failed = cached.get("failed_correction") or {}
    failed_result = read_json(previous / "bridge/results" / f"{CACHED_R4_6101_FAILED_CORRECTION['attempt_id']}.json")
    if (checksum(old_directory / "pause.request") != cached["previous_seed_pause_file_sha256"]
            or any(checksum(old_directory / name) != sha for name, sha in CACHED_R4_6101_BOUNDARY_FILES.items())
            or checksum(archived / "turn_00/attempt_00.terminal.json") != cached["archived_terminal_file_sha256"]
            or checksum(archived / "turn_00/decision.json") != cached["archived_decision_file_sha256"]
            or original_job.get("job_sha256") != cached["job_sha256"]
            or original_result.get("result_sha256") != cached["result_sha256"]
            or original_result.get("raw_response_sha256") != cached["raw_response_sha256"]
            or original_result.get("provider_dispatched") is not True
            or checksum(archived / "turn_01/attempt_00.terminal.json") != failed.get("terminal_file_sha256")
            or any(failed.get(key) != value for key, value in CACHED_R4_6101_FAILED_CORRECTION.items())
            or failed_result.get("result_sha256") != failed["result_sha256"]
            or failed_result.get("raw_response_sha256") is not None
            or any(path.is_file() for folder in ("outbox", "claims", "results") for path in (run / "bridge" / folder).glob("*.json"))):
        return None
    fit = _cached_r4_6101_fit(run / "seed_6101", cached, contract)
    imported = {"round": start, "fitted_labels": 493 if fit else None, "choice_count": 32,
                "removed_reference_fields": len(mirrors), "provider_dispatched": False,
                "archived_failed_correction": failed["error"], **(fit or {})}
    return run, contract, start, imported


def r5_after_6101_cached_r4(runs, previous, previous_contract, imported):
    if (previous_contract, imported.get("fitted_labels"), imported.get("checkpoint_sha256")) != (
            CACHED_R4_6101_CONTRACT, 493, CACHED_R4_6101_FITTED_CHECKPOINT):
        return None
    source = runs / f"{SERIAL_IPV4}-6101-relay-retry-08"
    run = runs / f"{SERIAL_IPV4}-6101-r5-continuation-01"
    approval = read_json(run / "authorization.json")
    cached = read_json(previous / "formal_contract.json")
    old = read_json(source / "formal_contract.json")
    worker = read_json(previous / "workers/seed_6101.json")
    if (fallback_contract(run) != CONTINUATION_6101_R5_CONTRACT
            or record_digest(approval) != CONTINUATION_6101_R5_AUTH
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or fallback_contract(source) != RELAY_6101_WAVE08_CONTRACT
            or (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels")) != ("complete", 5, 493)):
        return None
    stage = previous / "seed_6101/round_04"
    following = previous / "seed_6101/round_05"
    local_terminal = read_json(stage / "turn_00/attempt_00.terminal.json")
    local_id = local_terminal.get("attempt_id")
    fit = read_json(following / "transition.json").get("fit_path")
    if (not isinstance(fit, str) or not FIT.fullmatch(fit) or not isinstance(local_id, str) or not IDENTITY.fullmatch(local_id)
            or local_id in (CACHED_R4_6101_REVIEW["attempt_id"], CACHED_R4_6101_FAILED_CORRECTION["attempt_id"])):
        return None
    handoff = {"cached_selection_file_sha256": previous / "cached_response_import.json",
               "extension_source_sha256": following / "source.json",
               "extension_fit_audit_sha256": following / fit / "fit_audit.json",
               "extension_freeze_sha256": stage / "batch_freeze.json"}
    if any(checksum(path) != CONTINUATION_6101_R5_HANDOFF[key] for key, path in handoff.items()):
        return None
    custody = cached["predecessor"]["cached_selection"]
    original = old.get("predecessor") or {}
    record = read_json(run / "formal_contract.json")
    profile, proof = record.get("profile", {}), record.get("predecessor") or {}
    boundaries = profile.get("resume_boundaries", {})
    descriptor = boundaries.get("6101", {})
    expected = {**CONTINUATION_6101_R5_HANDOFF, "dispatcher_module": "dispatch_v13_6101_r5_continuation",
                "seed": 6101, "round": 5, "parent_run_directory": str(source),
                "parent_contract_sha256": RELAY_6101_WAVE08_CONTRACT, "parent_authorization_sha256": RELAY_6101_WAVE08_AUTH,
                "parent_seed_pause_sha256": CACHED_R4_6101_REVIEW["previous_seed_pause_file_sha256"],
                "extension_run_directory": str(previous), "extension_contract_sha256": CACHED_R4_6101_CONTRACT,
                "extension_authorization_sha256": CACHED_R4_6101_AUTH, "cached_local_attempt_id": local_id,
                "cached_local_terminal_sha256": CACHED_R4_6101_LOCAL_TERMINAL,
                "cached_selection_sha256": record_digest(custody), "reviewed_cached_selection": custody,
                "extension_transition_sha256": CACHED_R4_6101_TRANSITION, "checkpoint_sha256": CACHED_R4_6101_FITTED_CHECKPOINT,
                "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
                "execution_budget": CONTINUATION_6101_R5_BUDGET}
    expected.update({key: original.get(key) for key in ("model_identity", "cache_directory", "public_resource_receipt_sha256", "bridge_wait_seconds")})
    expected_profile = {**old.get("profile", {}), "resume_boundaries": boundaries, "replay_eligible_from_round": 5,
                        "sealed_relay_retry_seconds": RELAY_RETRY_SECONDS, "audited_future_transport_replay": True}
    if (record.get("budget") != CONTINUATION_6101_R5_BUDGET
            or record.get("source_sha256") != {**cached.get("source_sha256", {}), **CONTINUATION_6101_R5_SOURCES}
            or record_digest({key: proof.get(key) for key in expected}) != record_digest(expected)
            or profile != expected_profile or set(boundaries) != {"6101"} or profile.get("target_labels") != TARGET_LABELS
            or set(descriptor) != {"round", "state_sha256", "memory_sha256"} or descriptor.get("round") != 5
            or descriptor.get("state_sha256") != record_digest(read_json(following / "source.json"))
            or descriptor.get("memory_sha256") != record_digest(read_json(stage / "memory.json"))):
        return None
    return run, CONTINUATION_6101_R5_CONTRACT, 5, profile["ipv4_readiness_windows"]


def relay_retry_07_6101_contract(runs, previous, previous_contract):
    seed = 6101
    if previous_contract != CACHED_R2_6101_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-6101-relay-retry-07"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="cached-r2-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old = read_json(previous / "formal_contract.json")
    old_profile, old_proof = old.get("profile", {}), old.get("predecessor") or {}
    current = read_json(run / "formal_contract.json")
    boundaries = profile.get("resume_boundaries", {})
    exact = {**RELAY_6101_WAVE07_PROOF, "previous_authorization_sha256": CACHED_R2_6101_AUTH,
             "reviewed_fitted_prefix_files": RELAY_6101_WAVE07_FITTED_FILES, "checkpoint_sha256": CACHED_R2_6101_CHECKPOINT,
             "reviewed_cached_import": RELAY_6101_WAVE07_CACHED_REVIEW,
             "public_resource_receipt_sha256": RELAY_6101_WAVE07_RESOURCE_SHA,
             "previous_transport_observation": RELAY_6101_WAVE07_OBSERVATION}
    if (contract != RELAY_6101_WAVE07_CONTRACT or record_digest(read_json(run / "authorization.json")) != RELAY_6101_WAVE07_AUTH
            or start != 3 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 2
            or set(boundaries) != {str(seed)} or profile != {**old_profile, "resume_boundaries": boundaries,
                                                           "replay_eligible_from_round": 4}
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("sse_inactivity_seconds") != 900
            or profile.get("request_timeouts", {}).get("total_seconds") != 3600 or "cached_import" in proof
            or old.get("budget") != CACHED_R2_6101_BUDGET
            or old_proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
            or current.get("budget") != RELAY_6101_WAVE07_PROOF["execution_budget"]
            or any(current.get("source_sha256", {}).get(name) != sha for name, sha in RELAY_12203_WAVE12_SOURCES.items())
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    cached = old_proof.get("cached_import") or {}
    review = RELAY_6101_WAVE07_CACHED_REVIEW
    if (cached.get("provider_post") is not False or record_digest(cached) != review["proof_sha256"]
            or checksum(previous / "cached_response_import.json") != review["receipt_file_sha256"]):
        return None
    directory = previous / "seed_6101"
    turn = directory / "round_03/turn_00"
    if {path.name for path in turn.glob("attempt_*.json")} != {"attempt_00.started.json", "attempt_00.terminal.json"}:
        return None
    receipts = _reviewed_fitted_relay_receipts(directory, start, proof, RELAY_6101_WAVE07_FITTED_FILES,
                                              (RELAY_6101_WAVE07_PRIOR_ATTEMPT,), (proof["previous_terminal_file_sha256"],))
    if receipts is None or proof.get("reviewed_retry_chain") != [
        _relay_review(previous, previous_contract, CACHED_R2_6101_AUTH, proof, receipts),
    ]:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def relay_retry_07_157_contract(runs, previous, previous_contract):
    seed = 157
    if previous_contract != CACHED_R3_157_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-07"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="cached-r3-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old = read_json(previous / "formal_contract.json")
    old_profile, old_proof = old.get("profile", {}), old.get("predecessor") or {}
    boundaries = profile.get("resume_boundaries", {})
    expected_profile = {**old_profile, "resume_boundaries": boundaries, "replay_eligible_from_round": 5}
    exact = {**RELAY_157_WAVE07_PROOF, "previous_authorization_sha256": CACHED_R3_157_AUTH,
             "reviewed_fitted_prefix_files": CACHED_R3_157_FITTED_FILES,
             "reviewed_cached_import": RELAY_157_WAVE07_CACHED_REVIEW,
             "public_resource_receipt_sha256": RELAY_157_WAVE07_RESOURCE_SHA,
             "previous_transport_observation": RELAY_157_WAVE07_OBSERVATION}
    record = read_json(run / "formal_contract.json")
    sources = record.get("source_sha256") or {}
    if (contract != RELAY_157_WAVE07_CONTRACT or record_digest(read_json(run / "authorization.json")) != RELAY_157_WAVE07_AUTH
            or start != 4 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 3
            or set(boundaries) != {str(seed)} or record_digest(profile) != record_digest(expected_profile)
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("sse_inactivity_seconds") != 900
            or record.get("budget") != RELAY_157_WAVE07_PROOF["execution_budget"]
            or "cached_import" in proof or any(sources.get(name) != sha for name, sha in RELAY_RETRY_QUEUE_SOURCES.items())
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    cached = old_proof.get("cached_import") or {}
    if (cached.get("provider_post") is not False or record_digest(cached) != RELAY_157_WAVE07_CACHED_REVIEW["proof_sha256"]
            or checksum(previous / "cached_response_import.json") != RELAY_157_WAVE07_CACHED_REVIEW["receipt_file_sha256"]):
        return None
    directory = previous / f"seed_{seed}"
    receipts = _reviewed_fitted_relay_receipts(directory, start, proof, CACHED_R3_157_FITTED_FILES,
                                              (RELAY_157_WAVE07_PRIOR_ATTEMPT,), (proof["previous_terminal_file_sha256"],))
    if (receipts is None or (directory / "round_04/turn_00/attempt_01.started.json").exists()
            or proof.get("reviewed_retry_chain") != [_relay_review(previous, previous_contract, CACHED_R3_157_AUTH, proof, receipts)]):
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def cached_r7_contract(runs, seed, previous, previous_contract):
    if seed != 887:
        return None
    run = runs / f"{SERIAL_IPV4}-887-cached-r7-01"
    if candidate := _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_887_cached_r7", 0):
        contract, start, profile, proof = candidate
        cached = proof.get("cached_import") or {}
        generation = read_json(run / "coordinator_generation.json")
        if (start == 7 and profile.get("replay_eligible_from_round") == 8
                and profile.get("audited_future_transport_replay") is True
                and profile.get("ipv4_readiness_windows") == 3
                and cached.get("provider_post") is False and cached.get("normalized_plan_sha256") == CACHED_R7_PLAN_SHA
                and read_json(run / "cached_response_import.json") == cached
                and pid(generation.get("pid"))
                and isinstance(identifier := generation.get("id"), str) and IDENTITY.fullmatch(identifier)):
            return run, contract, start, profile["ipv4_readiness_windows"]
    return None


def cached_r8_relay_contract(runs, seed, previous, previous_contract):
    if seed != 887 or previous_contract != RELAY_887_CACHED_CONTRACT:
        return None
    # Wave 01 stopped at source verification before dispatch. Wave 02 is a direct cached-run successor.
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-02"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="cached-r7-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old = read_json(previous / "formal_contract.json")
    old_profile, old_proof = old.get("profile", {}), old.get("predecessor") or {}
    boundaries = profile.get("resume_boundaries", {})
    exact = {**RELAY_887_R8_PROOF, "previous_authorization_sha256": RELAY_887_CACHED_AUTH,
             "previous_transport_observation": RELAY_887_R8_OBSERVATION,
             "reviewed_fitted_prefix_files": RELAY_887_FITTED_PREFIX, "reviewed_cached_import": RELAY_887_CACHED_REVIEW,
             "public_resource_receipt_sha256": RELAY_887_RESOURCE_SHA}
    expected_profile = {**old_profile, "resume_boundaries": boundaries, "replay_eligible_from_round": 9,
                        "sealed_relay_retry_seconds": RELAY_RETRY_SECONDS}
    sources = read_json(run / "formal_contract.json").get("source_sha256") or {}
    if (start != 8 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 7
            or set(boundaries) != {str(seed)} or record_digest(profile) != record_digest(expected_profile)
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or "cached_import" in proof
            or any(sources.get(name) != sha for name, sha in RELAY_887_R8_WAVE02_SOURCES.items())
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    cached = old_proof.get("cached_import") or {}
    cache_receipt = previous / "cached_response_import.json"
    if (cached.get("provider_post") is not False or record_digest(cached) != RELAY_887_CACHED_REVIEW["proof_sha256"]
            or checksum(cache_receipt) != RELAY_887_CACHED_REVIEW["receipt_file_sha256"] or read_json(cache_receipt) != cached
            or old_proof.get("previous_contract_sha256") != RELAY_887_CACHED_REVIEW["resource_contract_sha256"]
            or old_proof.get("previous_authorization_sha256") != RELAY_887_CACHED_REVIEW["resource_authorization_sha256"]):
        return None
    directory = previous / f"seed_{seed}"
    cached_terminal = read_json(directory / "round_07/turn_00/attempt_00.terminal.json")
    identity = cached_terminal.get("attempt_id")
    if (cached_terminal.get("status") != "completed" or cached_terminal.get("provider_dispatched") is not False
            or not isinstance(identity, str) or not IDENTITY.fullmatch(identity)
            or (previous / "bridge/outbox" / f"{identity}.json").exists()):
        return None
    receipts = _reviewed_fitted_relay_receipts(directory, start, proof, RELAY_887_FITTED_PREFIX,
                                              (RELAY_887_R8_ATTEMPT,), (proof["previous_terminal_file_sha256"],))
    if receipts is None or proof.get("reviewed_retry_chain") != [_relay_review(previous, previous_contract,
                                                                                RELAY_887_CACHED_AUTH, proof, receipts)]:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def r8_targeted_repair(runs, seed, previous, previous_contract):
    if (seed != 887 or previous.name != f"{SERIAL_IPV4}-887-relay-retry-04" or previous_contract != REPAIR_887_PARENT):
        return None
    run = runs / REPAIR_887_RUN
    approval = read_json(run / "authorization.json")
    record = read_json(run / "formal_contract.json")
    profile, proof = record.get("profile", {}), record.get("predecessor") or {}
    if (fallback_contract(run) != REPAIR_887_CONTRACT or record_digest(approval) != REPAIR_887_AUTH
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or proof.get("dispatcher_module") != "dispatch_v13_887_r8_targeted_repair"
            or proof.get("previous_run_directory") != str(previous) or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("prior_response_reposted") is not False
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
            or profile.get("confirmation_seeds") != [887] or profile.get("target_labels") != 621
            or profile.get("resume_boundaries", {}).get("887", {}).get("round") != 8
            or profile.get("ipv4_readiness_windows") != 3):
        return None
    return run, REPAIR_887_CONTRACT, 8, 3


def r8_6101_targeted_repair(runs, previous, previous_contract):
    if previous_contract != RELAY_6101_WAVE28_CONTRACT or fallback_contract(previous) != previous_contract:
        return None
    run = runs / REPAIR_6101_R8_RUN
    candidate = _signed_recovery_successor(
        run, 6101, previous, previous_contract, "dispatch_v13_6101_r8_targeted_repair", 1,
        previous_suffix="relay-retry-28", target_labels=621,
    )
    if candidate is None:
        return None
    contract, start, profile, proof = candidate
    signed = read_json(run / "formal_contract.json")
    budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    metadata = profile.get("fixed_reference_repair")
    metadata_file = run / "fixed_reference_repair.json"
    transport = profile.get("transport") or {}
    parent_stage = previous / "seed_6101/round_08"
    if (contract != REPAIR_6101_R8_CONTRACT
            or record_digest(read_json(run / "authorization.json")) != REPAIR_6101_R8_AUTH
            or proof.get("previous_authorization_sha256") != RELAY_6101_WAVE28_AUTH
            or start != 8 or profile.get("ipv4_readiness_windows") != 3
            or proof.get("prior_response_reposted") is not False or proof.get("logical_index") != 0
            or signed.get("budget") != budget or proof.get("execution_budget") != budget
            or (transport.get("logical_replies"), transport.get("transport_attempts")) != (1, 1)
            or proof.get("fixed_reference_repair_sha256") != REPAIR_6101_R8_METADATA
            or record_digest(metadata) != REPAIR_6101_R8_METADATA or read_json(metadata_file) != metadata
            or checksum(metadata_file) != REPAIR_6101_R8_METADATA_FILE
            or any(signed.get("source_sha256", {}).get(k) != v for k, v in REPAIR_6101_R8_SOURCES.items())
            or any(checksum(previous / name) != sha for name, sha in REPAIR_6101_R8_PARENT_FILES.items())
            or set(parent_stage.glob("turn_*/attempt_*.started.json"))
                != {parent_stage / f"turn_00/attempt_{number:02d}.started.json" for number in (0, 1)}
            or any((parent_stage / name).exists() for name in ("selection.json", "batch_freeze.json"))):
        return None
    directory = run / "seed_6101"
    if (directory / "round_09/transition.json").exists() and (
            latest_round(directory)[1].get("checkpoint_sha256") != REPAIR_6101_R8_CHECKPOINT
            or checksum(directory / "round_08/batch_freeze.json") != REPAIR_6101_R8_FREEZE):
        return None
    return run, contract, start, profile["ipv4_readiness_windows"]


def r9_after_6101_r8_repair(runs, previous, previous_contract):
    if previous.name != REPAIR_6101_R8_RUN or previous_contract != REPAIR_6101_R8_CONTRACT:
        return None
    run = runs / CONTINUATION_6101_R9_RUN
    signed, approval = (read_json(run / name) for name in ("formal_contract.json", "authorization.json"))
    proof, profile = signed.get("predecessor") or {}, signed.get("profile") or {}
    transport, handoff = profile.get("transport") or {}, proof.get("r8_handoff") or {}
    fit_files, bridge = handoff.get("fit_files") or {}, handoff.get("bridge") or {}
    original = runs / f"{SERIAL_IPV4}-6101-relay-retry-28"
    required = {
        "dispatcher_module": "dispatch_v13_6101_r9_continuation", "seed": 6101, "round": 9,
        "previous_run_directory": str(previous), "previous_contract_sha256": previous_contract,
        "previous_authorization_sha256": REPAIR_6101_R8_AUTH, "normal_contract_sha256": RELAY_6101_WAVE28_CONTRACT,
        "normal_authorization_sha256": RELAY_6101_WAVE28_AUTH, "normal_run_directory": str(original),
        "checkpoint_sha256": REPAIR_6101_R8_CHECKPOINT, "recovery_budget": CONTINUATION_6101_R9_SCOPE,
        "execution_budget": CONTINUATION_6101_R9_BUDGET,
    }
    attempt = bridge.get("attempt_id")
    if (fallback_contract(previous) != previous_contract
            or record_digest(read_json(previous / "authorization.json")) != REPAIR_6101_R8_AUTH
            or fallback_contract(run) != CONTINUATION_6101_R9_CONTRACT
            or record_digest(approval) != CONTINUATION_6101_R9_AUTH
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)
            or any(proof.get(k) != v for k, v in required.items())
            or record_digest(handoff) != CONTINUATION_6101_R9_HANDOFF
            or record_digest(proof.get("r9_request")) != CONTINUATION_6101_R9_REQUEST
            or not fit_files or any(checksum(previous / name) != sha for name, sha in fit_files.items())
            or bridge.get("provider_post") is not True or not isinstance(attempt, str) or not IDENTITY.fullmatch(attempt)
            or checksum(previous / "bridge/outbox" / f"{attempt}.json") != bridge.get("job_file_sha256")
            or checksum(previous / "bridge/results" / f"{attempt}.json") != bridge.get("result_file_sha256")
            or checksum(previous / "seed_6101/round_08/batch_freeze.json") != REPAIR_6101_R8_FREEZE
            or latest_round(previous / "seed_6101")[1].get("checkpoint_sha256") != REPAIR_6101_R8_CHECKPOINT
            or profile.get("resume_boundaries") != {"6101": CONTINUATION_6101_R9_DESCRIPTOR}
            or profile.get("confirmation_seeds") != [6101] or profile.get("target_labels") != 1485
            or profile.get("full_imported_boundary_round") != 9 or profile.get("replay_eligible_from_round") != 9
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or "fixed_reference_repair" in profile
            or (transport.get("logical_replies"), transport.get("transport_attempts")) != (3, 2)
            or signed.get("budget") != CONTINUATION_6101_R9_BUDGET
            or any(signed.get("source_sha256", {}).get(k) != v for k, v in CONTINUATION_6101_R9_SOURCES.items())):
        return None
    return run, CONTINUATION_6101_R9_CONTRACT, 9, profile["ipv4_readiness_windows"]


def r10_cached_after_6101_r9(runs, previous, previous_contract):
    if previous.name != CONTINUATION_6101_R9_RUN or previous_contract != CONTINUATION_6101_R9_CONTRACT:
        return None
    run = runs / CACHED_R10_6101_RUN
    signed = _signed_retry_record(run, CACHED_R10_6101_CONTRACT, CACHED_R10_6101_AUTH)
    if (signed is None or _signed_retry_record(previous, previous_contract, CONTINUATION_6101_R9_AUTH) is None
            or latest_round(previous / "seed_6101")[1].get("checkpoint_sha256") != CACHED_R10_6101_START):
        return None
    record, _ = signed
    profile, proof = record.get("profile") or {}, record.get("predecessor") or {}
    custody = proof.get("cached_selection") or {}
    directory = run / "seed_6101"
    stage = directory / "round_10"
    worker = read_json(run / "workers/seed_6101.json")
    terminal = read_json(stage / "turn_00/attempt_00.terminal.json")
    freeze = read_json(stage / "batch_freeze.json")
    dataset_ids = freeze.get("batch_ids")
    choices = read_json(stage / "selection.json").get("choices")
    latest, metrics = latest_round(directory)
    expected = {
        "dispatcher_module": "run_v13_6101_r10_cached", "seed": 6101, "round": 10,
        "previous_run_directory": str(previous), "previous_contract_sha256": previous_contract,
        "previous_authorization_sha256": CONTINUATION_6101_R9_AUTH, "checkpoint_sha256": CACHED_R10_6101_START,
        "execution_budget": CACHED_R10_6101_BUDGET,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
    }
    if (any(proof.get(key) != value for key, value in expected.items()) or record.get("budget") != CACHED_R10_6101_BUDGET
            or profile.get("confirmation_seeds") != [6101] or profile.get("target_labels") != 685
            or profile.get("resume_boundaries", {}).get("6101", {}).get("round") != 10
            or (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels")) != ("complete", 11, 685)
            or (latest, metrics.get("checkpoint_sha256")) != (11, CACHED_R10_6101_FIT)
            or any(checksum(run / name) != sha for name, sha in CACHED_R10_6101_FILES.items())
            or custody.get("provider_post") is not False or record_digest(custody) != CACHED_R10_6101_RECEIPT
            or read_json(run / "cached_response_import.json") != custody
            or (terminal.get("status"), terminal.get("provider_dispatched"), terminal.get("cached_selection_sha256"),
                terminal.get("imported_from_response_id")) != ("completed", False, CACHED_R10_6101_RECEIPT, custody.get("response_id"))
            or freeze.get("request_id") != terminal.get("request_id")
            or type(dataset_ids) is not list or len(dataset_ids) != 32 or len(set(dataset_ids)) != 32
            or type(choices) is not list or len(choices) != 32 or any(not isinstance(choice, dict) for choice in choices)
            or record_digest([choice.get("candidate_id") for choice in choices]) != custody.get("ordered_candidate_ids_sha256")
            or any(any((run / "bridge" / folder).glob("*.json")) for folder in ("outbox", "results"))):
        return None
    return run, CACHED_R10_6101_CONTRACT, 10, {
        "round": 10, "fitted_labels": 685, "choice_count": 32, "removed_reference_fields": 0,
        "provider_dispatched": False, "archived_failed_correction": custody["original_failure"]["code"],
    }


def r11_6101_after_cached_r10(runs, previous, previous_contract):
    # Reached only after r10_cached_after_6101_r9 has verified the freeze, zero-POST terminal and immutable fit.
    if previous.name != CACHED_R10_6101_RUN or previous_contract != CACHED_R10_6101_CONTRACT:
        return None
    run = runs / CONTINUATION_6101_R11_RUN
    signed = _signed_retry_record(run, CONTINUATION_6101_R11_CONTRACT, CONTINUATION_6101_R11_AUTH)
    if signed is None:
        return None
    record, _ = signed
    profile, proof = record.get("profile") or {}, record.get("predecessor") or {}
    transport = profile.get("transport") or {}
    expected = {
        "dispatcher_module": "dispatch_v13_6101_r11_continuation", "seed": 6101, "round": 11,
        "previous_run_directory": str(previous), "previous_contract_sha256": previous_contract,
        "previous_authorization_sha256": CACHED_R10_6101_AUTH,
        "normal_run_directory": str(runs / CONTINUATION_6101_R9_RUN),
        "normal_contract_sha256": CONTINUATION_6101_R9_CONTRACT, "normal_authorization_sha256": CONTINUATION_6101_R9_AUTH,
        "checkpoint_sha256": CACHED_R10_6101_FIT, "execution_budget": CONTINUATION_6101_R11_BUDGET,
        "recovery_budget": CONTINUATION_6101_R9_SCOPE,
        "cached_response_import_file_sha256": CACHED_R10_6101_FILES["cached_response_import.json"],
    }
    if (any(proof.get(key) != value for key, value in expected.items())
            or record.get("budget") != CONTINUATION_6101_R11_BUDGET
            or any(record.get("source_sha256", {}).get(key) != sha for key, sha in CONTINUATION_6101_R11_SOURCES.items())
            or record_digest(proof.get("r10_handoff")) != CONTINUATION_6101_R11_HANDOFF
            or record_digest(proof.get("r11_request")) != CONTINUATION_6101_R11_REQUEST
            or record_digest(proof.get("cached_selection")) != CACHED_R10_6101_RECEIPT
            or profile.get("resume_boundaries") != {"6101": CONTINUATION_6101_R11_DESCRIPTOR}
            or profile.get("confirmation_seeds") != [6101] or profile.get("target_labels") != TARGET_LABELS
            or profile.get("full_imported_boundary_round") != 11 or profile.get("replay_eligible_from_round") != 11
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or "fixed_reference_repair" in profile
            or (transport.get("logical_replies"), transport.get("transport_attempts")) != (3, 2)):
        return None
    return run, CONTINUATION_6101_R11_CONTRACT, 11, 3


def r33_after_887_targeted_repair(runs, previous, previous_contract):
    if previous_contract != REPAIR_887_R32_CONTRACT or fallback_contract(previous) != previous_contract:
        return None
    run = runs / CONTINUATION_887_R33_RUN
    signed, approval = (read_json(run / name) for name in ("formal_contract.json", "authorization.json"))
    proof, profile = signed.get("predecessor") or {}, signed.get("profile") or {}
    transport, handoff = profile.get("transport") or {}, proof.get("r32_handoff") or {}
    fit_files = handoff.get("fit_files") or {}
    previous_auth = record_digest(read_json(previous / "authorization.json"))
    required = {
        "dispatcher_module": "dispatch_v13_887_r33_continuation", "seed": 887, "round": 33,
        "previous_run_directory": str(previous), "previous_contract_sha256": previous_contract,
        "previous_authorization_sha256": REPAIR_887_R32_AUTH, "normal_contract_sha256": RELAY_887_WAVE11_CONTRACT,
        "normal_authorization_sha256": RELAY_887_WAVE11_AUTH, "checkpoint_sha256": REPAIR_887_R32_CHECKPOINT,
        "recovery_budget": CONTINUATION_887_R33_SCOPE, "execution_budget": CONTINUATION_887_R33_BUDGET,
    }
    if (fallback_contract(run) != CONTINUATION_887_R33_CONTRACT
            or record_digest(approval) != CONTINUATION_887_R33_AUTH or previous_auth != REPAIR_887_R32_AUTH
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)
            or any(proof.get(k) != v for k, v in required.items())
            or record_digest(handoff) != CONTINUATION_887_R33_HANDOFF
            or record_digest(proof.get("r33_request")) != CONTINUATION_887_R33_REQUEST
            or not fit_files or any(checksum(previous / name) != sha for name, sha in fit_files.items())
            or latest_round(previous / "seed_887")[1].get("checkpoint_sha256") != REPAIR_887_R32_CHECKPOINT
            or profile.get("resume_boundaries") != {"887": CONTINUATION_887_R33_DESCRIPTOR}
            or profile.get("confirmation_seeds") != [887] or profile.get("target_labels") != 1485
            or profile.get("full_imported_boundary_round") != 33 or profile.get("replay_eligible_from_round") != 33
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or "fixed_reference_repair" in profile
            or (transport.get("logical_replies"), transport.get("transport_attempts")) != (3, 2)
            or signed.get("budget") != CONTINUATION_887_R33_BUDGET
            or any(signed.get("source_sha256", {}).get(k) != v for k, v in CONTINUATION_887_R33_SOURCES.items())):
        return None
    return run, CONTINUATION_887_R33_CONTRACT, 33, profile["ipv4_readiness_windows"]


def r32_targeted_repair(runs, previous, previous_contract):
    if previous_contract != RELAY_887_WAVE11_CONTRACT:
        return None
    run = runs / REPAIR_887_R32_RUN
    candidate = _signed_recovery_successor(
        run, 887, previous, previous_contract, "dispatch_v13_887_r32_targeted_repair", 1,
        previous_suffix="relay-retry-11", target_labels=1389,
    )
    if candidate is None:
        return None
    contract, start, profile, proof = candidate
    signed, approval = (read_json(run / name) for name in ("formal_contract.json", "authorization.json"))
    original = previous / "seed_887/round_32"
    turn = original / "turn_00"
    budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    metadata = profile.get("fixed_reference_repair")
    metadata_file = run / "fixed_reference_repair.json"
    dispatch_policy = profile.get("transport", {})
    if (contract != REPAIR_887_R32_CONTRACT or record_digest(approval) != REPAIR_887_R32_AUTH
            or fallback_contract(previous) != previous_contract
            or proof.get("previous_authorization_sha256") != RELAY_887_WAVE11_AUTH
            or start != 32 or profile.get("ipv4_readiness_windows") != 3
            or profile.get("resume_boundaries") != {"887": RELAY_887_WAVE11_DESCRIPTOR}
            or proof.get("prior_response_reposted") is not False or proof.get("logical_index") != 0
            or signed.get("budget") != budget or proof.get("execution_budget") != budget
            or (dispatch_policy.get("logical_replies"), dispatch_policy.get("transport_attempts")) != (1, 1)
            or proof.get("fixed_reference_repair_sha256") != REPAIR_887_R32_METADATA_SHA
            or record_digest(metadata) != REPAIR_887_R32_METADATA_SHA or read_json(metadata_file) != metadata
            or checksum(metadata_file) != REPAIR_887_R32_METADATA_FILE
            or any(signed.get("source_sha256", {}).get(k) != v for k, v in REPAIR_887_R32_SOURCES.items())
            or any(checksum(previous / name) != sha for name, sha in REPAIR_887_R32_ORIGINAL.items())
            or set(original.glob("turn_*/attempt_*.started.json")) != {turn / "attempt_00.started.json"}
            or any((original / name).exists() for name in ("selection.json", "batch_freeze.json"))):
        return None
    seed_directory = run / "seed_887"
    if (seed_directory / "round_33/transition.json").exists() and (
        latest_round(seed_directory)[1].get("checkpoint_sha256") != REPAIR_887_R32_CHECKPOINT
        or any(checksum(seed_directory / name) != sha for name, sha in REPAIR_887_R32_FITTED.items())
    ):
        return None
    return run, contract, start, profile["ipv4_readiness_windows"]


def r13_targeted_repair(runs, previous, previous_contract):
    if previous_contract != RELAY_887_WAVE10_CONTRACT:
        return None
    run = runs / REPAIR_887_R13_RUN
    candidate = _signed_recovery_successor(
        run, 887, previous, previous_contract, "dispatch_v13_887_r13_targeted_repair", 1,
        previous_suffix="relay-retry-10", target_labels=781,
    )
    if candidate is None:
        return None
    contract, start, profile, proof = candidate
    signed = read_json(run / "formal_contract.json")
    prior = read_json(previous / "formal_contract.json")
    metadata = profile.get("fixed_reference_repair")
    approval = read_json(run / "authorization.json")
    repair_budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    original = previous / "seed_887/round_13"
    turn = original / "turn_00"
    if (contract != REPAIR_887_R13_CONTRACT or record_digest(approval) != REPAIR_887_R13_AUTH
            or fallback_contract(previous) != previous_contract
            or proof.get("previous_authorization_sha256") != RELAY_887_WAVE10_AUTH
            or start != 13 or profile.get("ipv4_readiness_windows") != 3
            or profile["resume_boundaries"] != prior.get("profile", {}).get("resume_boundaries")
            or proof.get("prior_response_reposted") is not False or proof.get("logical_index") != 0
            or signed.get("budget") != repair_budget or proof.get("execution_budget") != repair_budget
            or profile.get("transport", {}).get("logical_replies") != 1
            or profile["transport"].get("transport_attempts") != 1
            or proof.get("fixed_reference_repair_sha256") != REPAIR_887_R13_METADATA_SHA
            or record_digest(metadata) != REPAIR_887_R13_METADATA_SHA
            or read_json(run / "fixed_reference_repair.json") != metadata
            or any(signed.get("source_sha256", {}).get(name) != sha for name, sha in REPAIR_887_R13_SOURCES.items())
            or any(checksum(previous / name) != sha for name, sha in REPAIR_887_R13_ORIGINAL_RESPONSE_CUSTODY.items())
            or set(original.glob("turn_*/attempt_*.started.json")) != {turn / "attempt_00.started.json"}
            or any((original / name).exists() for name in ("selection.json", "batch_freeze.json"))
            or (original.with_name("round_14")).exists()):
        return None
    return run, contract, start, profile["ipv4_readiness_windows"]


def r13_cached_correction(runs, previous, previous_contract):
    if previous_contract != REPAIR_887_R13_CONTRACT:
        return None
    run = runs / CACHED_887_R13_RUN
    candidate = _signed_recovery_successor(
        run, 887, previous, previous_contract, "run_v13_887_r13_cached_correction", 0,
        previous_suffix="r13-targeted-repair-01", target_labels=781,
    )
    if candidate is None:
        return None
    contract, start, _, proof = candidate
    approval = read_json(run / "authorization.json")
    record = read_json(run / "formal_contract.json")
    custody = proof.get("cached_correction") or {}
    terminal = read_json(run / "seed_887/round_13/turn_00/attempt_00.terminal.json")
    zero_budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
    if (contract != CACHED_887_R13_CONTRACT or record_digest(approval) != CACHED_887_R13_AUTH or start != 13
            or proof.get("previous_authorization_sha256") != REPAIR_887_R13_AUTH
            or record.get("budget") != zero_budget or proof.get("execution_budget") != zero_budget
            or custody.get("provider_post") is not False or record_digest(custody) != CACHED_887_R13_CUSTODY_SHA
            or custody.get("projected_plan_sha256") != CACHED_887_R13_PROJECTED
            or read_json(run / "cached_correction.json") != custody
            or terminal.get("status") != "completed" or terminal.get("provider_dispatched") is not False
            or terminal.get("usage") != {} or terminal.get("attempt_id") != CACHED_887_R13_ATTEMPT
            or any(checksum(previous / name) != sha for name, sha in CACHED_887_R13_PAID_FILES.items())
            or any(checksum(run / name) != sha for name, sha in CACHED_887_R13_FILES.items())
            or any(any((run / "bridge" / folder).glob("*.json")) for folder in ("outbox", "claims", "results"))):
        return None
    return run, contract, start, 0


def r14_after_887_cached_correction(runs, previous, previous_contract):
    if previous.name != CACHED_887_R13_RUN or previous_contract != CACHED_887_R13_CONTRACT:
        return None
    run = runs / CONTINUATION_887_R14_RUN
    record = read_json(run / "formal_contract.json")
    approval = read_json(run / "authorization.json")
    profile, proof = record.get("profile", {}), record.get("predecessor") or {}
    hashes = CACHED_887_R13_FILES
    expected = {
        "dispatcher_module": "dispatch_v13_887_r14_continuation", "seed": 887, "round": 14,
        "parent_run_directory": str(runs / f"{SERIAL_IPV4}-887-relay-retry-10"),
        "parent_contract_sha256": RELAY_887_WAVE10_CONTRACT, "parent_authorization_sha256": RELAY_887_WAVE10_AUTH,
        "paid_correction_contract_sha256": REPAIR_887_R13_CONTRACT, "paid_correction_authorization_sha256": REPAIR_887_R13_AUTH,
        "extension_run_directory": str(previous), "extension_contract_sha256": CACHED_887_R13_CONTRACT,
        "extension_authorization_sha256": CACHED_887_R13_AUTH, "checkpoint_sha256": CACHED_887_R13_CHECKPOINT,
        "cached_local_attempt_id": CACHED_887_R13_ATTEMPT, "cached_correction_sha256": CACHED_887_R13_CUSTODY_SHA,
        "cached_correction_file_sha256": hashes["cached_correction.json"],
        "cached_local_terminal_sha256": hashes["seed_887/round_13/turn_00/attempt_00.terminal.json"],
        "extension_freeze_sha256": hashes["seed_887/round_13/batch_freeze.json"],
        "extension_source_sha256": hashes["seed_887/round_14/source.json"],
        "extension_transition_sha256": hashes["seed_887/round_14/transition.json"],
        "extension_fit_audit_sha256": hashes["seed_887/round_14/fit_000/fit_audit.json"],
        "extension_projected_plan_sha256": CACHED_887_R13_PROJECTED,
        "execution_budget": CONTINUATION_887_R14_BUDGET,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
    }
    descriptor = {"round": 14, "state_sha256": CACHED_887_R13_STATE_SHA, "memory_sha256": CACHED_887_R13_MEMORY_SHA}
    if (fallback_contract(run) != CONTINUATION_887_R14_CONTRACT or record_digest(approval) != CONTINUATION_887_R14_AUTH
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or any(proof.get(key) != value for key, value in expected.items())
            or record.get("budget") != CONTINUATION_887_R14_BUDGET
            or profile.get("resume_boundaries") != {"887": descriptor} or profile.get("target_labels") != TARGET_LABELS
            or profile.get("confirmation_seeds") != [887] or "fixed_reference_repair" in profile
            or profile.get("replay_eligible_from_round") != 14 or profile.get("audited_future_transport_replay") is not True
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or (proof.get("r14_request") or {}).get("payload_sha256") != CONTINUATION_887_R14_PAYLOAD
            or (proof.get("r14_request") or {}).get("checkpoint_sha256") != CACHED_887_R13_CHECKPOINT
            or any(record.get("source_sha256", {}).get(name) != sha for name, sha in CONTINUATION_887_R14_SOURCES.items())):
        return None
    return run, CONTINUATION_887_R14_CONTRACT, 14, 3


def r19_cached_correction(runs, previous, previous_contract):
    handoff = CACHED_887_R19_HANDOFF
    if previous_contract != CONTINUATION_887_R14_CONTRACT:
        return None
    run = runs / CACHED_887_R19_RUN
    candidate = _signed_recovery_successor(
        run, 887, previous, previous_contract, "run_v13_887_r19_cached", 0,
        previous_suffix="r14-continuation-01", target_labels=973,
    )
    if candidate is None:
        return None
    contract, start, _, proof = candidate
    record = read_json(run / "formal_contract.json")
    custody = proof.get("cached_selection") or {}
    terminal = read_json(run / "seed_887/round_19/turn_00/attempt_00.terminal.json")
    files = CACHED_887_R19_FILES
    worker = read_json(run / "workers/seed_887.json")
    zero_budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
    if (contract != handoff["contract_sha256"] or record_digest(read_json(run / "authorization.json")) != handoff["authorization_sha256"]
            or start != 19 or proof.get("previous_authorization_sha256") != CONTINUATION_887_R14_AUTH
            or record.get("budget") != zero_budget or proof.get("execution_budget") != zero_budget
            or custody.get("provider_post") is not False or record_digest(custody) != handoff["cached_selection_sha256"]
            or read_json(run / "cached_response_import.json") != custody
            or terminal.get("status") != "completed" or terminal.get("provider_dispatched") is not False
            or terminal.get("usage") != {} or terminal.get("attempt_id") != handoff["local_attempt_id"]
            or (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels")) != ("complete", 20, 973)
            or any(checksum(run / name) != sha for name, sha in files.items())
            or any(any((run / "bridge" / folder).glob("*.json")) for folder in ("outbox", "claims", "results"))):
        return None
    return run, contract, start, 0


def r20_after_887_cached_correction(runs, previous, previous_contract):
    handoff = CACHED_887_R19_HANDOFF
    if previous.name != CACHED_887_R19_RUN or previous_contract != handoff["contract_sha256"]:
        return None
    run = runs / CONTINUATION_887_R20_RUN
    record = read_json(run / "formal_contract.json")
    approval = read_json(run / "authorization.json")
    profile, proof = record.get("profile", {}), record.get("predecessor") or {}
    expected = {
        "dispatcher_module": "dispatch_v13_887_r20_continuation", "seed": 887, "round": 20,
        "parent_run_directory": str(runs / CONTINUATION_887_R14_RUN),
        "parent_contract_sha256": CONTINUATION_887_R14_CONTRACT, "parent_authorization_sha256": CONTINUATION_887_R14_AUTH,
        "extension_run_directory": str(previous), "checkpoint_sha256": handoff["checkpoint_sha256"],
        "r19_handoff": handoff, "cached_response_import_file_sha256": CACHED_887_R19_FILES["cached_response_import.json"],
        "r20_request": CONTINUATION_887_R20_REQUEST, "execution_budget": CONTINUATION_887_R20_BUDGET,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
    }
    descriptor = {"round": 20, "state_sha256": handoff["state_sha256"], "memory_sha256": handoff["scientific_memory_sha256"]}
    sources = record.get("source_sha256", {})
    if (fallback_contract(run) != CONTINUATION_887_R20_CONTRACT or record_digest(approval) != CONTINUATION_887_R20_AUTH
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or any(proof.get(key) != value for key, value in expected.items())
            or record_digest(proof.get("cached_selection")) != handoff["cached_selection_sha256"]
            or record.get("budget") != CONTINUATION_887_R20_BUDGET or profile.get("resume_boundaries") != {"887": descriptor}
            or profile.get("target_labels") != TARGET_LABELS or profile.get("confirmation_seeds") != [887]
            or "fixed_reference_repair" in profile or profile.get("replay_eligible_from_round") != 20
            or profile.get("audited_future_transport_replay") is not True
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or sources.get("studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py") != CONTINUATION_887_R20_PARSER
            or sources.get("studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r19_cached.py")
            != CONTINUATION_887_R20_CACHED_HELPER
            or any(sources.get(name) != sha for name, sha in CONTINUATION_887_R20_SOURCES.items())):
        return None
    return run, CONTINUATION_887_R20_CONTRACT, 20, 3


def r8_cached_correction(runs, previous, previous_contract):
    if previous.name != REPAIR_887_RUN or previous_contract != REPAIR_887_CONTRACT:
        return None
    run = runs / CACHED_887_R8_RUN
    approval = read_json(run / "authorization.json")
    record = read_json(run / "formal_contract.json")
    profile, proof = record.get("profile", {}), record.get("predecessor") or {}
    custody = proof.get("cached_correction") or {}
    receipt = run / "cached_correction.json"
    if (fallback_contract(run) != CACHED_887_R8_CONTRACT or record_digest(approval) != CACHED_887_R8_AUTH
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or proof.get("dispatcher_module") != "run_v13_887_r8_cached_correction"
            or proof.get("previous_run_directory") != str(previous) or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("previous_authorization_sha256") != REPAIR_887_AUTH
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
            or custody.get("provider_post") is not False or custody.get("projected_plan_sha256") != CACHED_887_R8_PROJECTION
            or checksum(receipt) != CACHED_887_R8_RECEIPT or read_json(receipt) != custody
            or profile.get("confirmation_seeds") != [887] or profile.get("target_labels") != 621
            or profile.get("resume_boundaries", {}).get("887", {}).get("round") != 8):
        return None
    return run, CACHED_887_R8_CONTRACT, 8, 0


def r9_after_cached_correction(runs, previous, previous_contract):
    following = previous / "seed_887/round_09"
    latest, metrics = latest_round(previous / "seed_887")
    if (previous_contract != CACHED_887_R8_CONTRACT or latest != 9
            or metrics.get("checkpoint_sha256") != CACHED_887_R8_BOUNDARY["checkpoint_sha256"]
            or checksum(following / "source.json") != CACHED_887_R8_BOUNDARY["extension_source_sha256"]
            or checksum(following / "transition.json") != CACHED_887_R8_BOUNDARY["extension_transition_sha256"]):
        return None
    run = runs / CONTINUATION_887_R9_RUN
    record = read_json(run / "formal_contract.json")
    approval = read_json(run / "authorization.json")
    profile, proof = record.get("profile", {}), record.get("predecessor") or {}
    expected = {**CACHED_887_R8_BOUNDARY, "dispatcher_module": "dispatch_v13_887_r9_continuation",
                "seed": 887, "round": 9, "extension_run_directory": str(previous),
                "extension_contract_sha256": CACHED_887_R8_CONTRACT, "extension_authorization_sha256": CACHED_887_R8_AUTH,
                "paid_correction_contract_sha256": REPAIR_887_CONTRACT, "paid_correction_authorization_sha256": REPAIR_887_AUTH,
                "extension_projected_plan_sha256": CACHED_887_R8_PROJECTION, "cached_correction_file_sha256": CACHED_887_R8_RECEIPT,
                "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
                "execution_budget": CONTINUATION_887_R9_BUDGET}
    sources = record.get("source_sha256", {})
    transport = profile.get("transport", {})
    if (fallback_contract(run) != CONTINUATION_887_R9_CONTRACT or record_digest(approval) != CONTINUATION_887_R9_AUTH
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or record_digest({key: proof.get(key) for key in expected}) != record_digest(expected)
            or any(sources.get(name) != sha for name, sha in CONTINUATION_887_R9_SOURCES.items())
            or profile.get("confirmation_seeds") != [887] or profile.get("target_labels") != TARGET_LABELS
            or profile.get("resume_boundaries", {}).get("887", {}).get("round") != 9
            or profile.get("ipv4_readiness_windows") != 3 or "fixed_reference_repair" in profile
            or profile.get("replay_eligible_from_round") != 9 or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS
            or (transport.get("requested_model"), transport.get("reasoning_effort"), transport.get("sdk_version"),
                transport.get("logical_replies"), transport.get("transport_attempts")) != ("gpt-6.2-astra", "xhigh", "2.28.0", 3, 2)):
        return None
    return run, CONTINUATION_887_R9_CONTRACT, 9, 3


def relay_887_r12_after_r9(runs, previous, previous_contract):
    seed = 887
    if previous_contract != CONTINUATION_887_R9_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-887-relay-retry-05"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="r9-continuation-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old = read_json(previous / "formal_contract.json")
    old_profile, old_proof = old.get("profile", {}), old.get("predecessor") or {}
    signed = read_json(run / "formal_contract.json")
    sources = signed.get("source_sha256") or {}
    boundaries = profile.get("resume_boundaries", {})
    expected_profile = {**old_profile, "resume_boundaries": boundaries, "replay_eligible_from_round": 13}
    former_budget = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    exact = {**RELAY_887_WAVE05_PROOF, "previous_authorization_sha256": CONTINUATION_887_R9_AUTH,
             "reviewed_fitted_prefix_files": RELAY_887_R12_FITTED_FILES, "checkpoint_sha256": RELAY_887_R12_CHECKPOINT,
             "previous_transport_observation": RELAY_887_WAVE05_OBSERVATION}
    if (contract != RELAY_887_WAVE05_CONTRACT or record_digest(read_json(run / "authorization.json")) != RELAY_887_WAVE05_AUTH
            or start != 12 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 9
            or set(boundaries) != {str(seed)} or record_digest(profile) != record_digest(expected_profile)
            or profile.get("audited_future_transport_replay") is not True or profile.get("ipv4_readiness_windows") != 3
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS or profile.get("sse_inactivity_seconds") != 900
            or profile.get("request_timeouts", {}).get("total_seconds") != 3600
            or old.get("budget") != CONTINUATION_887_R9_BUDGET or old_proof.get("recovery_budget") != former_budget
            or signed.get("budget") != RELAY_887_WAVE05_PROOF["execution_budget"]
            or any(sources.get(name) != sha for name, sha in RELAY_2357_WAVE05_SOURCES.items())
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    directory = previous / "seed_887"
    turn = directory / "round_12/turn_00"
    latest, metrics = latest_round(directory)
    if ((latest, metrics.get("checkpoint_sha256")) != (12, RELAY_887_R12_CHECKPOINT)
            or {path.name for path in turn.glob("attempt_*.json")} != {"attempt_00.started.json", "attempt_00.terminal.json"}):
        return None
    attempts = _reviewed_fitted_relay_receipts(directory, start, proof, RELAY_887_R12_FITTED_FILES,
                                              (RELAY_887_R12_ATTEMPT,), (proof["previous_terminal_file_sha256"],))
    if attempts is None or proof.get("reviewed_retry_chain") != [
        _relay_review(previous, previous_contract, CONTINUATION_887_R9_AUTH, proof, attempts),
    ]:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def _plain_filesystem_path(path):
    try:
        return path.resolve(strict=True) == path
    except (OSError, RuntimeError):
        return False


def _relay_pin(value):
    match = re.fullmatch(r"(157|887|2357|6101|12203):([1-9][0-9]?):([0-9a-f]{64}):([0-9a-f]{64})", value)
    if match:
        seed, wave, contract, approval = match.groups()
        return int(seed), int(wave), contract, approval
    return None


def discover_relay_pins(private, ssh_host):
    """Read launch arguments as data. Ship only seed, wave and two expected digests."""
    directory = private / "launchd"
    if not _plain_filesystem_path(directory):
        return []
    launcher = re.compile(r"com\.purellm\.a7-ipv4-v13-(157|887|2357|6101|12203)-relay-retry-([0-9]{2})\.plist")
    module = "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_v13_retry_successor"
    required = {"--seed", "--wave", "--remote-run-dir", "--remote-root", "--contract-sha256", "--authorization-sha256"}
    pins = []
    try:
        with os.scandir(directory) as records:
            paths = sorted(entry.name for entry in records if launcher.fullmatch(entry.name))
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return []
    try:
        directory_status = os.fstat(descriptor)
        if directory_status.st_uid != os.getuid() or directory_status.st_mode & 0o022:
            return []
        for name in paths:
            suffix = name.removeprefix("com.purellm.").removesuffix(".plist")
            seed, wave = launcher.fullmatch(name).groups()
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
                with os.fdopen(fd, "rb") as stream:
                    status = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid()
                            or status.st_mode & 0o022 or status.st_size > 65536):
                        continue
                    plist = plistlib.loads(stream.read(65537))
            except (OSError, ValueError, TypeError, OverflowError, ExpatError, plistlib.InvalidFileException):
                continue
            args = plist.get("ProgramArguments") if isinstance(plist, dict) else None
            if (not isinstance(args, list) or len(args) < 8 or any(not isinstance(value, str) for value in args)
                    or plist.get("Label") != "com.purellm." + suffix or args[:2] != ["/usr/bin/caffeinate", "-i"]
                    or args[3:8] != ["-B", "-u", "-m", module, "--execute-paid"] or len(args) % 2):
                continue
            flags = dict(zip(args[8::2], args[9::2], strict=True))
            if (set(flags) not in (required, required | {"--ssh-host"})
                    or len(flags) * 2 != len(args) - 8 or flags.get("--ssh-host", "eit-hpc") != ssh_host
                    or flags["--seed"] != seed or flags["--wave"] != str(int(wave))
                    or flags["--remote-run-dir"] != str(RUNS / f"{SERIAL_IPV4}-{seed}-relay-retry-{wave}")
                    or not re.fullmatch(str(RUNS.parent / "releases" / f"ipv4-v13-{seed}-relay-retry-{wave}-")
                                        + r"[0-9a-f]{64}", flags["--remote-root"])):
                continue
            pin = f'{seed}:{int(wave)}:{flags["--contract-sha256"]}:{flags["--authorization-sha256"]}'
            if _relay_pin(pin):
                pins.append(pin)
    finally:
        os.close(descriptor)
    return pins


def _signed_retry_record(run, contract, authorization=None):
    contract_path, approval_path = (run / name for name in ("formal_contract.json", "authorization.json"))
    if not all(_plain_filesystem_path(path) and path.is_file() for path in (contract_path, approval_path)):
        return None
    record, approval = read_json(contract_path), read_json(approval_path)
    approval_sha = record_digest(approval)
    if (record.get("run_directory") != str(run) or record_digest(record) != contract
            or approval.get("contract_sha256") != contract
            or any(approval.get(flag) is not True for flag in FORMAL_APPROVAL_FLAGS)
            or authorization is not None and approval_sha != authorization):
        return None
    return record, approval_sha


def _owned_relay_retry_evidence(previous, seed, start, proof, previous_contract, previous_auth):
    directory = previous / f"seed_{seed}"
    active = directory / f"round_{start:02d}"
    turn = active / "turn_00"
    pause = directory / "pause.request"
    coordinator = previous / "coordinator_generation.json"
    worker = previous / "workers" / f"seed_{seed}.json"
    receipts = proof.get("previous_attempts")
    if (not all(_plain_filesystem_path(file) and file.is_file() for file in (pause, coordinator, worker))
            or live(pid(read_json(coordinator).get("pid"))) or live(pid(read_json(worker).get("pid")))
            or read_json(worker).get("status") != "failed"
            or type(receipts) is not list or not 1 <= len(receipts) <= 2
            or (directory / f"round_{start+1:02d}").exists()
            or any((active / name).exists() for name in ("selection.json", "batch_freeze.json", "memory.json"))
            or set(active.glob("turn_*/attempt_*.started.json"))
                != {turn / f"attempt_{index:02d}.started.json" for index in range(len(receipts))}):
        return False
    parent_request_path = turn / "request.json"
    if not _plain_filesystem_path(parent_request_path):
        return False
    request = read_json(parent_request_path)
    request_id, payload = proof.get("previous_request_id"), proof.get("payload_sha256")
    formal = request.get("binding", {}).get("a7_formal", {})
    if (not all(isinstance(value, str) and SHA.fullmatch(value) for value in (request_id, payload))
            or request.get("request_id") != request_id or request.get("payload_sha256") != payload
            or record_digest(request.get("payload")) != payload or request.get("logical_index") != 0
            or (formal.get("contract_sha256"), formal.get("authorization_sha256"), formal.get("seed"),
                formal.get("mock")) != (previous_contract, previous_auth, seed, False)):
        return False
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            return False
        attempt = receipt.get("attempt_id")
        if not isinstance(attempt, str) or not IDENTITY.fullmatch(attempt) or receipt.get("attempt") != index:
            return False
        terminal_path = turn / f"attempt_{index:02d}.terminal.json"
        started_path = turn / f"attempt_{index:02d}.started.json"
        job_path = previous / "bridge/outbox" / f"{attempt}.json"
        result_path = previous / "bridge/results" / f"{attempt}.json"
        if not all(_plain_filesystem_path(p) for p in (started_path, terminal_path, job_path, result_path)):
            return False
        started, terminal, job, result = map(read_json, (started_path, terminal_path, job_path, result_path))
        error = "response_id_mismatch"
        failure = result.get("outcome", {}).get("failure", {})
        identity = failure.get("reply_identity", {})
        provider_request = identity.get("provider_request_id")
        if (not isinstance(provider_request, str) or not re.fullmatch(r"(?i)[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", provider_request)
                or identity.get("response_id") != "resp_" + provider_request.replace("-", "").lower()
                or identity.get("provider_status") != "failed" or identity.get("usage") != {}
                or (receipt.get("error"), receipt.get("provider_dispatched"), receipt.get("ambiguous")) != (error, True, True)
                or (started.get("attempt_id"), started.get("attempt")) != (attempt, index)
                or (terminal.get("attempt_id"), terminal.get("request_id"), terminal.get("status"), terminal.get("error"),
                    terminal.get("provider_dispatched"), terminal.get("ambiguous")) != (attempt, request_id, "stopped", error, True, True)
                or receipt.get("terminal_sha256") != record_digest(terminal)
                or (job.get("attempt_id"), job.get("request_id"), job.get("request")) != (attempt, request_id, request)
                or job.get("job_sha256") != receipt.get("job_sha256")
                or record_digest({key: value for key, value in job.items() if key != "job_sha256"}) != receipt.get("job_sha256")
                or (result.get("attempt_id"), result.get("request_id"), result.get("job_sha256"), result.get("result_sha256"))
                    != (attempt, request_id, receipt.get("job_sha256"), receipt.get("result_sha256"))
                or result.get("provider_dispatched") is not True or result.get("raw_response") is not None
                or (failure.get("code"), failure.get("ambiguous")) != (error, True)
                or record_digest({key: value for key, value in result.items() if key != "result_sha256"}) != receipt.get("result_sha256")):
            return False
    not_before = seconds(terminal.get("not_before_unix"))
    failed = seconds(terminal.get("failed_unix"))
    observation = proof.get("previous_transport_observation") or {}
    return (checksum(terminal_path) == proof.get("previous_terminal_file_sha256")
            and not_before is not None and failed is not None
            and observation.get("event_type") in ("response.output_item.added", "response.output_item.done")
            and observation.get("http_status") == 200 and observation.get("conflicting_event_type") == "response.failed"
            and observation.get("provider_request_id") == provider_request
            and observation.get("conflicting_response_id") == identity["response_id"]
            and observation.get("error_type") == "response_id_mismatch"
            and observation.get("response_id_matches_transport_alias") is True
            and observation.get("provider_error_code") == "upstream_error"
            and proof.get("relay_cooldown_clock") == "not_before_unix" and not_before >= failed
            and proof.get("relay_not_before_unix") == not_before + RELAY_RETRY_SECONDS)


def discovered_relay_successor(runs, previous, previous_contract, seed, wave, pin):
    """Display only contiguous, locally pinned, same-science relay renewals."""
    if previous.name != f"{SERIAL_IPV4}-{seed}-relay-retry-{wave-1:02d}":
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-{wave:02d}"
    old = _signed_retry_record(previous, previous_contract)
    current = _signed_retry_record(run, *pin)
    if old is None or current is None:
        return None
    (old_record, old_auth), (record, _) = old, current
    profile, proof = record.get("profile") or {}, record.get("predecessor") or {}
    old_profile, old_proof = old_record.get("profile") or {}, old_record.get("predecessor") or {}
    start = proof.get("round")
    physical = proof.get("newly_authorized_boundary_physical_attempts")
    if (type(start) is not int or not 0 <= start < TARGET_ROUNDS or type(physical) is not int or physical not in (1, 2)
            or old_proof.get("round") != start or profile.get("target_labels") != TARGET_LABELS
            or profile.get("confirmation_seeds") != [seed] or profile.get("ipv4_readiness_windows") != 3
            or profile.get("audited_future_transport_replay") is not True
            or profile.get("sealed_relay_retry_seconds") != RELAY_RETRY_SECONDS):
        return None
    budget = {"first_logical_requests": TARGET_ROUNDS - start, "maximum_logical_requests": (TARGET_ROUNDS-start-1)*3+1,
              "maximum_physical_attempts": (TARGET_ROUNDS-start-1)*6+physical}
    kept_proof = ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256", "payload_sha256",
                  "model_identity", "cache_directory", "public_resource_receipt_sha256", "bridge_wait_seconds")
    retained_record = ("api_execution", "gpu_slots", "asset_sha256", "anchor_root", "lease_root", "initial_policy")
    old_sources, sources = old_record.get("source_sha256"), record.get("source_sha256")
    producer = "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py"
    if (not isinstance(sources, dict) or not isinstance(old_sources, dict) or producer not in old_sources
            or not isinstance(sources.get(producer), str) or not SHA.fullmatch(sources[producer])
            or sources != {**old_sources, producer: sources[producer]}
            or any(key not in old_record or record.get(key) != old_record[key] for key in retained_record)
            or any(key not in old_proof or proof.get(key) != old_proof[key] for key in kept_proof)
            or proof.get("dispatcher_module") != "dispatch_v13_retry_successor"
            or old_proof.get("dispatcher_module") != proof["dispatcher_module"]
            or (proof.get("seed"), proof.get("logical_index"), proof.get("previous_contract_sha256"),
                proof.get("previous_authorization_sha256"), proof.get("previous_run_directory")) !=
                (seed, 0, previous_contract, old_auth, str(previous))
            or profile != {**old_profile, "replay_eligible_from_round": start + (physical == 1)}
            or record.get("budget") != budget or proof.get("execution_budget") != budget
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": physical}
            or type(old_proof.get("reviewed_retry_chain")) is not list
            or not _owned_relay_retry_evidence(previous, seed, start, proof, previous_contract, old_auth)
            or proof.get("reviewed_retry_chain") != [*old_proof["reviewed_retry_chain"], _relay_review(
                previous, previous_contract, old_auth, proof, proof["previous_attempts"])]):
        return None
    return run, pin[0], start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"], proof


def remote_seed(runs, seed, relay_pins=None):
    cohort_contract = None
    for name, mode, dispatcher, capability in (
        (SERIAL_IPV4, "ipv4", "dispatch_prepared_ipv4", "gated_ipv4"),
        (BACKGROUND, "background", "dispatch_prepared_background", "verified_route"),
    ):
        cohort = runs / name
        if candidate := fallback_contract(cohort):
            record = read_json(cohort / "formal_contract.json")
            profile = record.get("profile", {})
            if (tuple(profile.get("confirmation_seeds", ())) == SEEDS
                    and profile.get("provider_capability") == capability
                    and (record.get("predecessor") or {}).get("dispatcher_module") == dispatcher):
                cohort_contract = candidate
                break
    direct_full = runs / f"{DIRECT_FULL}-{seed}"
    full = runs / f"{FULL}-{seed}"
    direct = runs / f"{DIRECT_887 if seed == 887 else DIRECT}-{seed}"
    continuation = runs / f"{CONTINUATION}-{seed}"
    fallback = runs / (FIRST_887 if seed == 887 else ORIGINAL)
    repair = runs / f"{REPAIR}-{seed}"
    direct_full_contract, full_contract, repair_contract, direct_contract, continuation_contract = (
        dispatch_contract(run, seed) for run in (direct_full, full, repair, direct, continuation))
    extension = runs / f"{SERIAL_IPV4}-{seed}-extension-01"
    extension_dispatcher = IPV4_EXTENSION_DISPATCHERS.get(seed)
    extension_record = read_json(extension / "formal_contract.json") if extension_dispatcher else {}
    extension_parent = extension_record.get("predecessor") or {}
    extension_contract = (
        fallback_contract(extension) if cohort_contract and mode == "ipv4"
        and extension_parent.get("parent_contract_sha256") == cohort_contract
        and extension_parent.get("dispatcher_module") == extension_dispatcher
        and extension_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    follow_on = runs / f"{SERIAL_IPV4}-{seed}-continuation-01"
    follow_on_record = read_json(follow_on / "formal_contract.json") if extension_contract else {}
    follow_on_parent = follow_on_record.get("predecessor") or {}
    follow_on_contract = (
        fallback_contract(follow_on) if extension_contract
        and follow_on_parent.get("extension_contract_sha256") == extension_contract
        and follow_on_parent.get("parent_contract_sha256") == cohort_contract
        and follow_on_parent.get("dispatcher_module") == IPV4_CONTINUATION_DISPATCHERS.get(seed)
        and follow_on_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    r2_extension = runs / IPV4_R2_EXTENSION
    r2_record = read_json(r2_extension / "formal_contract.json") if seed == 2357 and follow_on_contract else {}
    r2_parent = r2_record.get("predecessor") or {}
    r2_contract = (
        fallback_contract(r2_extension) if follow_on_contract
        and r2_parent.get("parent_contract_sha256") == follow_on_contract
        and r2_parent.get("dispatcher_module") == "dispatch_v13_2357_r2_sse_extension"
        and r2_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    r2_continuation = runs / IPV4_R2_CONTINUATION
    r2_follow_record = read_json(r2_continuation / "formal_contract.json") if r2_contract else {}
    r2_follow_prior = r2_follow_record.get("predecessor") or {}
    r2_follow_contract = (
        fallback_contract(r2_continuation) if r2_contract
        and r2_follow_prior.get("parent_contract_sha256") == follow_on_contract
        and r2_follow_prior.get("extension_contract_sha256") == r2_contract
        and r2_follow_prior.get("dispatcher_module") == "dispatch_v13_2357_r2_continuation"
        and r2_follow_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    extra_name, extra_dispatcher, extra_segment = {
        887: (IPV4_JSON_EXTENSION, "dispatch_v13_887_json_extension", "ipv4_json_extension"),
        6101: (IPV4_WS_EXTENSION, "dispatch_v13_6101_ws_extension", "ipv4_ws_extension"),
    }.get(seed, ("", "", ""))
    extra = runs / extra_name
    extra_record = read_json(extra / "formal_contract.json") if extra_name and extension_contract else {}
    extra_prior = extra_record.get("predecessor") or {}
    extra_contract = (
        fallback_contract(extra) if extension_contract
        and extra_prior.get("parent_contract_sha256") == cohort_contract
        and extra_prior.get("previous_contract_sha256") == extension_contract
        and extra_prior.get("dispatcher_module") == extra_dispatcher
        and extra_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    if extra_contract:
        run, contract, segment = extra, extra_contract, extra_segment
        ancestors = (extension, cohort, direct_full, continuation, fallback)
    elif r2_follow_contract:
        run, contract, segment = r2_continuation, r2_follow_contract, "ipv4_r2_continuation"
        ancestors = (r2_extension, follow_on, extension, cohort, direct_full, continuation, fallback)
    elif r2_contract:
        run, contract, segment = r2_extension, r2_contract, "ipv4_r2_extension"
        ancestors = (follow_on, extension, cohort, direct_full, continuation, fallback)
    elif follow_on_contract:
        run, contract, segment = follow_on, follow_on_contract, "ipv4_continuation"
        ancestors = (extension, cohort, direct_full, continuation, fallback)
    elif extension_contract:
        run, contract, segment = extension, extension_contract, "ipv4_extension"
        ancestors = (cohort, direct_full, continuation, fallback)
    elif cohort_contract:
        run, contract, segment = cohort, cohort_contract, mode
        ancestors = tuple(path for path, approved in (
            (direct_full, direct_full_contract), (repair, repair_contract), (direct, direct_contract),
            (continuation, continuation_contract), (fallback, fallback_contract(fallback)),
        ) if approved)
    elif direct_full_contract:
        run, contract, segment = direct_full, direct_full_contract, "v11"
        ancestors = tuple(path for path, approved in (
            (repair, repair_contract), (direct, direct_contract), (continuation, continuation_contract),
            (fallback, fallback_contract(fallback)),
        ) if approved)
    elif full_contract:
        run, contract, segment = full, full_contract, "v6"
        ancestors = (repair, direct, continuation, fallback)
    elif repair_contract:
        run, contract, segment = repair, repair_contract, "repair"
        ancestors = (direct, continuation, fallback)
    elif direct_contract:
        run, contract, segment = direct, direct_contract, "direct"
        ancestors = (continuation, fallback)
    elif continuation_contract:
        run, contract, segment = continuation, continuation_contract, "v5"
        ancestors = (fallback,)
    else:
        run, contract, segment = fallback, fallback_contract(fallback), "historical"
        ancestors = ()
    bounded = runs / f"{SERIAL_IPV4}-{seed}-bounded-sse-01"
    bounded_record = read_json(bounded / "formal_contract.json") if contract and seed in BOUNDED_SSE_END_ROUNDS else {}
    bounded_prior = bounded_record.get("predecessor") or {}
    if (bounded_prior.get("previous_contract_sha256") == contract
            and bounded_prior.get("previous_run_directory") == str(run)
            and bounded_prior.get("dispatcher_module") == "dispatch_v13_bounded_sse_recovery"
            and bounded_record.get("profile", {}).get("confirmation_seeds") == [seed]
            and (bounded_contract := fallback_contract(bounded))):
        ancestors = (run, *ancestors)
        run, contract, segment = bounded, bounded_contract, "ipv4_bounded_sse"
    recovery_round, readiness_windows = None, 0
    relay_retry = relay_boundary_not_before = None
    cache_import = {}
    if mode == "ipv4" and (resilient := resilient_contract(
        runs, seed, run, contract, cohort_contract, follow_on_contract,
    )):
        ancestors = (run, *ancestors)
        run, contract, recovery_round, readiness_windows = resilient
        segment = "ipv4_resilient_sse"
        if recovered := resilient_failure_contract(runs, seed, run, contract, recovery_round):
            ancestors = (run, *ancestors)
            run, contract = recovered
            segment = "ipv4_resilient_failure"
            if cached := cached_r7_contract(runs, seed, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows = cached
                segment = "ipv4_cached_r7"
                if successor := cached_r8_relay_contract(runs, seed, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                    segment = "ipv4_relay_retry_02"
                    relay_retry = RELAY_RETRY_SECONDS
            elif relay := relay_retry_contract(runs, seed, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows = relay
                segment = "ipv4_relay_retry"
                relay_retry = RELAY_RETRY_SECONDS
                if wave02 := relay_retry_02_contract(runs, seed, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = wave02
                    segment = "ipv4_relay_retry_02"
            if segment == "ipv4_relay_retry_02":
                wave = 3
                while successor := reviewed_relay_contract(runs, seed, wave, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                    segment = f"ipv4_relay_retry_{wave:02d}"
                    wave += 1
                if targeted := r8_targeted_repair(runs, seed, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows = targeted
                    segment = "ipv4_r8_targeted_repair"
                    relay_retry = relay_boundary_not_before = None
                    if imported := r8_cached_correction(runs, run, contract):
                        ancestors = (run, *ancestors)
                        run, contract, recovery_round, readiness_windows = imported
                        segment = "ipv4_r8_cached_correction"
                        if continued := r9_after_cached_correction(runs, run, contract):
                            ancestors = (run, *ancestors)
                            run, contract, recovery_round, readiness_windows = continued
                            segment = "ipv4_887_r9_continuation"
                            relay_retry = RELAY_RETRY_SECONDS
                            if successor := relay_887_r12_after_r9(runs, run, contract):
                                ancestors = (run, *ancestors)
                                run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                                segment = "ipv4_relay_retry_05"
                                wave = 6
                                while successor := reviewed_relay_contract(runs, seed, wave, run, contract):
                                    ancestors = (run, *ancestors)
                                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                                    segment = f"ipv4_relay_retry_{wave:02d}"
                                    wave += 1
            if segment == "ipv4_relay_retry_06" and (cached := cached_r3_157_contract(runs, seed, run, contract)):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows = cached
                segment = "ipv4_cached_r3"
                relay_retry = RELAY_RETRY_SECONDS
                relay_boundary_not_before = None
                if successor := relay_retry_07_157_contract(runs, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                    segment = "ipv4_relay_retry_07"
                    wave = 8
                    while successor := reviewed_relay_contract(runs, seed, wave, run, contract):
                        ancestors = (run, *ancestors)
                        run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                        segment = f"ipv4_relay_retry_{wave:02d}"
                        wave += 1
            if segment == "ipv4_relay_retry_06" and (cached := cached_r2_6101_contract(runs, seed, run, contract)):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows, cache_import = cached
                segment = "ipv4_cached_r2_6101"
                relay_retry = RELAY_RETRY_SECONDS
                relay_boundary_not_before = None
                if successor := relay_retry_07_6101_contract(runs, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                    segment = "ipv4_relay_retry_07"
                    cache_import["future_starts_at_round"] = recovery_round + 1
                    if successor := reviewed_relay_contract(runs, seed, 8, run, contract):
                        ancestors = (run, *ancestors)
                        run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                        segment = "ipv4_relay_retry_08"
                        if archived := cached_r4_6101_contract(runs, seed, run, contract):
                            ancestors = (run, *ancestors)
                            run, contract, recovery_round, cache_import = archived
                            segment = "ipv4_cached_r4_6101"
                            relay_retry = relay_boundary_not_before = None
                            if continued := r5_after_6101_cached_r4(runs, run, contract, cache_import):
                                ancestors = (run, *ancestors)
                                run, contract, recovery_round, readiness_windows = continued
                                segment = "ipv4_6101_r5_continuation"
                                relay_retry = RELAY_RETRY_SECONDS
                                wave = 9
                                while successor := reviewed_relay_contract(runs, seed, wave, run, contract):
                                    ancestors = (run, *ancestors)
                                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                                    segment = f"ipv4_relay_retry_{wave:02d}"
                                    wave += 1
    if seed == 6101 and (targeted := r8_6101_targeted_repair(runs, run, contract)):
        ancestors = (run, *ancestors)
        run, contract, recovery_round, readiness_windows = targeted
        segment = "ipv4_6101_r8_targeted_repair"
        relay_boundary_not_before = None
        if continued := r9_after_6101_r8_repair(runs, run, contract):
            ancestors = (run, *ancestors)
            run, contract, recovery_round, readiness_windows = continued
            segment = "ipv4_6101_r9_continuation"
            relay_retry = RELAY_RETRY_SECONDS
            if local := r10_cached_after_6101_r9(runs, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, cache_import = local
                segment = "ipv4_6101_r10_cached"
                relay_retry = relay_boundary_not_before = None
                if normal := r11_6101_after_cached_r10(runs, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows = normal
                    segment = "ipv4_6101_r11_continuation"
                    relay_retry = RELAY_RETRY_SECONDS
    if seed == 887 and (targeted := r13_targeted_repair(runs, run, contract)):
        ancestors = (run, *ancestors)
        run, contract, recovery_round, readiness_windows = targeted
        segment = "ipv4_887_r13_targeted_repair"
        relay_retry = relay_boundary_not_before = None
        if imported := r13_cached_correction(runs, run, contract):
            ancestors = (run, *ancestors)
            run, contract, recovery_round, readiness_windows = imported
            segment = "ipv4_887_r13_cached_correction"
            if continued := r14_after_887_cached_correction(runs, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows = continued
                segment = "ipv4_887_r14_continuation"
                relay_retry = RELAY_RETRY_SECONDS
    if seed == 887 and (imported := r19_cached_correction(runs, run, contract)):
        ancestors = (run, *ancestors)
        run, contract, recovery_round, readiness_windows = imported
        segment = "ipv4_887_r19_cached"
        relay_retry = relay_boundary_not_before = None
        if continued := r20_after_887_cached_correction(runs, run, contract):
            ancestors = (run, *ancestors)
            run, contract, recovery_round, readiness_windows = continued
            segment = "ipv4_887_r20_continuation"
            relay_retry = RELAY_RETRY_SECONDS
            if successor := reviewed_relay_contract(runs, seed, 11, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                segment = "ipv4_relay_retry_11"
                if targeted := r32_targeted_repair(runs, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows = targeted
                    segment = "ipv4_887_r32_targeted_repair"
                    relay_retry = relay_boundary_not_before = None
                    if continued := r33_after_887_targeted_repair(runs, run, contract):
                        ancestors = (run, *ancestors)
                        run, contract, recovery_round, readiness_windows = continued
                        segment = "ipv4_887_r33_continuation"
                        relay_retry = RELAY_RETRY_SECONDS
                        wave = 12
                        while successor := reviewed_relay_contract(runs, seed, wave, run, contract):
                            ancestors = (run, *ancestors)
                            run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                            segment = f"ipv4_relay_retry_{wave:02d}"
                            wave += 1
    dynamic_relay = None
    if relay_pins and (suffix := re.fullmatch(r"ipv4_relay_retry_([0-9]{2})", segment)):
        wave = int(suffix[1]) + 1
        while pin := relay_pins.get((seed, wave)):
            try:
                successor = discovered_relay_successor(runs, run, contract, seed, wave, pin)
            except (AttributeError, KeyError, OSError, TypeError, ValueError):
                successor = None
            if successor is None:
                break
            ancestors = (run, *ancestors)
            run, contract, recovery_round, readiness_windows, relay_boundary_not_before, dynamic_relay = successor
            segment = f"ipv4_relay_retry_{wave:02d}"
            relay_retry = RELAY_RETRY_SECONDS
            wave += 1
    if seed in UNSTARTED_ANCHORS and not cohort_contract and not any(
        (path / f"seed_{seed}").exists() for path in (run, *ancestors)
    ):
        anchor = (runs.parent.parent / "formal_cuda/row4g_b32_20261124/trajectories"
                  / f"random-{seed}/rounds/random/round_00/model/best.pt")
        expected = UNSTARTED_ANCHORS[seed]
        verified = checksum(anchor) == expected
        return {"seed": seed, "run": None, "contract_sha256": None, "segment": "anchor",
                "continuation": False, "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
                "completed_rounds": 0 if verified else None, "labels": 333 if verified else None,
                "phase": "l333_anchor_verified" if verified else "l333_anchor_unverified",
                "worker_pid": None, "worker_active": False, "worker_error": None, "coordinator_pid": None,
                "coordinator_active": False, "gpu": None, "api": {},
                "checkpoint_sha256": expected if verified else None}
    directory = run / f"seed_{seed}"
    latest, metrics = latest_round(directory)
    local_fit = segment == "ipv4_cached_r4_6101"
    local_887_fit = segment == "ipv4_887_r13_cached_correction"
    local_887_r19_fit = segment == "ipv4_887_r19_cached"
    if local_fit:
        latest = 5 if cache_import["fitted_labels"] == 493 else 4
        metrics = ({key: cache_import[key] for key in ("checkpoint_sha256", "validation_nrmse")} if latest == 5
                   else latest_round(ancestors[0] / "seed_6101")[1])
    for ancestor in ancestors:
        if latest is not None:
            break
        latest, metrics = latest_round(ancestor / f"seed_{seed}")
    if cohort_contract and latest is None:
        lane = record["predecessor"].get("lanes", {}).get(str(seed), {})
        start, checkpoint = lane.get("start_round"), lane.get("checkpoint_sha256")
        if type(start) is int and 0 <= start < TARGET_ROUNDS and isinstance(checkpoint, str) and SHA.fullmatch(checkpoint):
            latest, metrics = start, {"checkpoint_sha256": checkpoint}
    worker = read_json(run / "workers" / f"seed_{seed}.json")
    coordinator = pid(read_json(run / "coordinator_generation.json").get("pid"))
    coordinator_live = live(coordinator, run=run)
    worker_pid = pid(worker.get("pid"))
    worker_live = coordinator_live and (worker_pid == coordinator or live(worker_pid, coordinator=coordinator))
    paused = (run / "pause.request").is_file() or (directory / "pause.request").is_file()
    gpu = read_json(run / "gpu" / f"seed_{seed}.json")
    phase = safe_code(worker.get("status")) or "waiting_to_start"
    api = {} if local_fit or local_887_fit or local_887_r19_fit or segment == "ipv4_6101_r10_cached" else pending_attempt(run, directory, latest,
                                               failed=segment in RESILIENT_SEGMENTS and phase == "failed" and not worker_live,
                                               relay_retry=relay_retry, contract=contract)
    if segment in ("ipv4_cached_r7", "ipv4_cached_r3") and latest == recovery_round:
        if api.get("status") == "waiting_for_mac":
            api["status"] = "validating_cached_response"
        api["cached_import"] = True
    if (relay_boundary_not_before is not None and latest == recovery_round
            and not api and relay_boundary_not_before > time.time()):
        api = {"status": "relay_boundary_cooldown",
               "relay_retry_wait_seconds": max(0, math.ceil(relay_boundary_not_before - time.time()))}
    reviewed_previous_api = {}
    if seed == 2357 and segment in ("ipv4_relay_retry_02", "ipv4_relay_retry_04") and latest == recovery_round:
        telemetry = RELAY_2357_WAVE02_OBSERVATION if segment.endswith("_02") else RELAY_2357_WAVE04_OBSERVATION
        reviewed_previous_api = {name: telemetry[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api["complete_event_seen"] = False
        if segment.endswith("_04"):
            reviewed_previous_api["sse_inactivity_seconds"] = 900
    if (seed, segment) == (2357, "ipv4_relay_retry_05") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_2357_WAVE05_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, hard_deadline_seconds=3600)
    if seed == 2357 and segment in ("ipv4_relay_retry_06", "ipv4_relay_retry_07") and latest == recovery_round:
        telemetry = RELAY_2357_WAVE07_OBSERVATION if segment.endswith("_07") else RELAY_2357_WAVE06_OBSERVATION
        reviewed_previous_api = {name: telemetry[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if (seed, segment) == (2357, "ipv4_relay_retry_11") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_2357_WAVE11_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    alias_failure = {(2357, "ipv4_relay_retry_08"): RELAY_2357_WAVE08_OBSERVATION,
                     (2357, "ipv4_relay_retry_09"): RELAY_2357_WAVE09_OBSERVATION,
                     (2357, "ipv4_relay_retry_12"): RELAY_2357_WAVE12_OBSERVATION,
                     (2357, "ipv4_relay_retry_13"): RELAY_2357_WAVE13_OBSERVATION,
                     (2357, "ipv4_relay_retry_14"): RELAY_2357_WAVE14_OBSERVATION,
                     (6101, "ipv4_relay_retry_10"): RELAY_6101_WAVE10_OBSERVATION,
                     (6101, "ipv4_relay_retry_11"): RELAY_6101_WAVE11_OBSERVATION,
                     (6101, "ipv4_relay_retry_13"): RELAY_6101_WAVE13_OBSERVATION,
                     (157, "ipv4_relay_retry_14"): RELAY_157_WAVE14_OBSERVATION,
                     (2357, "ipv4_relay_retry_15"): RELAY_2357_WAVE15_OBSERVATION,
                     (12203, "ipv4_relay_retry_21"): RELAY_12203_WAVE21_OBSERVATION,
                     (6101, "ipv4_relay_retry_14"): RELAY_6101_WAVE14_OBSERVATION,
                     (157, "ipv4_relay_retry_15"): RELAY_157_WAVE15_OBSERVATION,
                     (157, "ipv4_relay_retry_16"): RELAY_157_WAVE16_OBSERVATION,
                     (2357, "ipv4_relay_retry_16"): RELAY_2357_WAVE16_OBSERVATION,
                     (6101, "ipv4_relay_retry_15"): RELAY_6101_WAVE15_OBSERVATION,
                     (12203, "ipv4_relay_retry_22"): RELAY_12203_WAVE22_OBSERVATION,
                     (2357, "ipv4_relay_retry_17"): RELAY_2357_WAVE17_OBSERVATION,
                     (12203, "ipv4_relay_retry_23"): RELAY_12203_WAVE23_OBSERVATION,
                     (6101, "ipv4_relay_retry_16"): RELAY_6101_WAVE16_OBSERVATION,
                     (157, "ipv4_relay_retry_17"): RELAY_157_WAVE17_OBSERVATION,
                     (2357, "ipv4_relay_retry_18"): RELAY_2357_WAVE18_OBSERVATION,
                     (12203, "ipv4_relay_retry_24"): RELAY_12203_WAVE24_OBSERVATION,
                     (6101, "ipv4_relay_retry_17"): RELAY_6101_WAVE17_OBSERVATION,
                     (157, "ipv4_relay_retry_18"): RELAY_157_WAVE18_OBSERVATION,
                     (157, "ipv4_relay_retry_19"): RELAY_157_WAVE19_OBSERVATION,
                     (6101, "ipv4_relay_retry_19"): RELAY_6101_WAVE19_OBSERVATION,
                     (12203, "ipv4_relay_retry_25"): RELAY_12203_WAVE25_OBSERVATION,
                     (12203, "ipv4_relay_retry_26"): RELAY_12203_WAVE26_OBSERVATION,
                     (157, "ipv4_relay_retry_20"): RELAY_157_WAVE20_OBSERVATION,
                     (2357, "ipv4_relay_retry_20"): RELAY_2357_WAVE20_OBSERVATION,
                     (6101, "ipv4_relay_retry_20"): RELAY_6101_WAVE20_OBSERVATION,
                     (12203, "ipv4_relay_retry_27"): RELAY_12203_WAVE27_OBSERVATION,
                     (157, "ipv4_relay_retry_21"): RELAY_157_WAVE21_OBSERVATION,
                     (157, "ipv4_relay_retry_22"): RELAY_157_WAVE22_OBSERVATION,
                     (157, "ipv4_relay_retry_23"): RELAY_157_WAVE23_OBSERVATION,
                     (157, "ipv4_relay_retry_24"): RELAY_157_WAVE24_OBSERVATION,
                     (157, "ipv4_relay_retry_25"): RELAY_157_WAVE25_OBSERVATION,
                     (157, "ipv4_relay_retry_26"): RELAY_157_WAVE26_OBSERVATION,
                     (157, "ipv4_relay_retry_27"): RELAY_157_WAVE27_OBSERVATION,
                     (157, "ipv4_relay_retry_28"): RELAY_157_WAVE28_OBSERVATION,
                     (157, "ipv4_relay_retry_29"): RELAY_157_WAVE29_OBSERVATION,
                     (157, "ipv4_relay_retry_30"): RELAY_157_WAVE30_OBSERVATION,
                     (157, "ipv4_relay_retry_31"): RELAY_157_WAVE31_OBSERVATION,
                     (2357, "ipv4_relay_retry_21"): RELAY_2357_WAVE21_OBSERVATION,
                     (2357, "ipv4_relay_retry_22"): RELAY_2357_WAVE22_OBSERVATION,
                     (2357, "ipv4_relay_retry_23"): RELAY_2357_WAVE23_OBSERVATION,
                     (2357, "ipv4_relay_retry_24"): RELAY_2357_WAVE24_OBSERVATION,
                     (2357, "ipv4_relay_retry_25"): RELAY_2357_WAVE25_OBSERVATION,
                     (2357, "ipv4_relay_retry_26"): RELAY_2357_WAVE26_OBSERVATION,
                     (2357, "ipv4_relay_retry_27"): RELAY_2357_WAVE27_OBSERVATION,
                     (2357, "ipv4_relay_retry_28"): RELAY_2357_WAVE28_OBSERVATION,
                     (2357, "ipv4_relay_retry_29"): RELAY_2357_WAVE29_OBSERVATION,
                     (2357, "ipv4_relay_retry_30"): RELAY_2357_WAVE30_OBSERVATION,
                     (2357, "ipv4_relay_retry_31"): RELAY_2357_WAVE31_OBSERVATION,
                     (6101, "ipv4_relay_retry_21"): RELAY_6101_WAVE21_OBSERVATION,
                     (6101, "ipv4_relay_retry_22"): RELAY_6101_WAVE22_OBSERVATION,
                     (6101, "ipv4_relay_retry_23"): RELAY_6101_WAVE23_OBSERVATION,
                     (6101, "ipv4_relay_retry_24"): RELAY_6101_WAVE24_OBSERVATION,
                     (6101, "ipv4_relay_retry_25"): RELAY_6101_WAVE25_OBSERVATION,
                     (6101, "ipv4_relay_retry_26"): RELAY_6101_WAVE26_OBSERVATION,
                     (6101, "ipv4_relay_retry_27"): RELAY_6101_WAVE27_OBSERVATION,
                     (6101, "ipv4_relay_retry_28"): RELAY_6101_WAVE28_OBSERVATION,
                     (12203, "ipv4_relay_retry_28"): RELAY_12203_WAVE28_OBSERVATION,
                     (12203, "ipv4_relay_retry_29"): RELAY_12203_WAVE29_OBSERVATION,
                     (12203, "ipv4_relay_retry_30"): RELAY_12203_WAVE30_OBSERVATION,
                     (12203, "ipv4_relay_retry_31"): RELAY_12203_WAVE31_OBSERVATION,
                     (12203, "ipv4_relay_retry_32"): RELAY_12203_WAVE32_OBSERVATION,
                     (12203, "ipv4_relay_retry_33"): RELAY_12203_WAVE33_OBSERVATION,
                     (12203, "ipv4_relay_retry_34"): RELAY_12203_WAVE34_OBSERVATION,
                     (12203, "ipv4_relay_retry_35"): RELAY_12203_WAVE35_OBSERVATION,
                     (12203, "ipv4_relay_retry_36"): RELAY_12203_WAVE36_OBSERVATION,
                     (12203, "ipv4_relay_retry_37"): RELAY_12203_WAVE37_OBSERVATION,
                     (887, "ipv4_relay_retry_12"): RELAY_887_WAVE12_OBSERVATION,
                     (887, "ipv4_relay_retry_13"): RELAY_887_WAVE13_OBSERVATION,
                     (887, "ipv4_relay_retry_14"): RELAY_887_WAVE14_OBSERVATION,
                     (887, "ipv4_relay_retry_15"): RELAY_887_WAVE15_OBSERVATION,
                     (12203, "ipv4_relay_retry_20"): RELAY_12203_WAVE20_OBSERVATION}.get((seed, segment))
    if alias_failure is not None and latest == recovery_round:
        observation = alias_failure
        reviewed_previous_api = {name: observation[name] for name in (
            "http_status", "error_type", "sse_event_count", "provider_error_code", "response_id_matches_transport_alias",
        ) if name in observation}
        reviewed_previous_api.update(complete_event_seen=False, event_type=observation.get("conflicting_event_type") or observation.get("event_type"))
        if (seed, segment) not in ((6101, "ipv4_relay_retry_13"), (6101, "ipv4_relay_retry_19")):
            reviewed_previous_api["archived_failure_count"] = 2
        archived_failure_sse_counts = {
            (2357, "ipv4_relay_retry_12"): RELAY_2357_WAVE12_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_13"): RELAY_2357_WAVE13_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_14"): RELAY_2357_WAVE14_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_10"): RELAY_6101_WAVE10_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_11"): RELAY_6101_WAVE11_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_20"): RELAY_12203_WAVE20_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_28"): RELAY_12203_WAVE28_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_29"): RELAY_12203_WAVE29_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_30"): RELAY_12203_WAVE30_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_31"): RELAY_12203_WAVE31_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_32"): RELAY_12203_WAVE32_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_33"): RELAY_12203_WAVE33_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_34"): RELAY_12203_WAVE34_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_35"): RELAY_12203_WAVE35_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_36"): RELAY_12203_WAVE36_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_37"): RELAY_12203_WAVE37_ARCHIVED_SSE_COUNTS,
            (887, "ipv4_relay_retry_12"): RELAY_887_WAVE12_ARCHIVED_SSE_COUNTS,
            (887, "ipv4_relay_retry_13"): RELAY_887_WAVE13_ARCHIVED_SSE_COUNTS,
            (887, "ipv4_relay_retry_14"): RELAY_887_WAVE14_ARCHIVED_SSE_COUNTS,
            (887, "ipv4_relay_retry_15"): RELAY_887_WAVE15_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_21"): RELAY_6101_WAVE21_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_22"): RELAY_6101_WAVE22_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_23"): RELAY_6101_WAVE23_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_24"): RELAY_6101_WAVE24_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_25"): RELAY_6101_WAVE25_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_26"): RELAY_6101_WAVE26_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_27"): RELAY_6101_WAVE27_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_28"): RELAY_6101_WAVE28_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_21"): RELAY_2357_WAVE21_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_22"): RELAY_2357_WAVE22_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_23"): RELAY_2357_WAVE23_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_24"): RELAY_2357_WAVE24_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_25"): RELAY_2357_WAVE25_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_26"): RELAY_2357_WAVE26_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_27"): RELAY_2357_WAVE27_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_28"): RELAY_2357_WAVE28_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_29"): RELAY_2357_WAVE29_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_30"): RELAY_2357_WAVE30_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_31"): RELAY_2357_WAVE31_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_21"): RELAY_157_WAVE21_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_22"): RELAY_157_WAVE22_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_23"): RELAY_157_WAVE23_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_24"): RELAY_157_WAVE24_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_25"): RELAY_157_WAVE25_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_26"): RELAY_157_WAVE26_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_27"): RELAY_157_WAVE27_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_28"): RELAY_157_WAVE28_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_29"): RELAY_157_WAVE29_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_30"): RELAY_157_WAVE30_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_31"): RELAY_157_WAVE31_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_27"): RELAY_12203_WAVE27_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_20"): RELAY_6101_WAVE20_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_20"): RELAY_2357_WAVE20_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_20"): RELAY_157_WAVE20_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_25"): RELAY_12203_WAVE25_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_26"): RELAY_12203_WAVE26_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_19"): RELAY_157_WAVE19_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_18"): RELAY_157_WAVE18_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_17"): RELAY_6101_WAVE17_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_24"): RELAY_12203_WAVE24_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_18"): RELAY_2357_WAVE18_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_17"): RELAY_157_WAVE17_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_16"): RELAY_6101_WAVE16_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_23"): RELAY_12203_WAVE23_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_17"): RELAY_2357_WAVE17_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_22"): RELAY_12203_WAVE22_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_15"): RELAY_6101_WAVE15_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_16"): RELAY_2357_WAVE16_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_16"): RELAY_157_WAVE16_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_15"): RELAY_157_WAVE15_ARCHIVED_SSE_COUNTS,
            (6101, "ipv4_relay_retry_14"): RELAY_6101_WAVE14_ARCHIVED_SSE_COUNTS,
            (12203, "ipv4_relay_retry_21"): RELAY_12203_WAVE21_ARCHIVED_SSE_COUNTS,
            (2357, "ipv4_relay_retry_15"): RELAY_2357_WAVE15_ARCHIVED_SSE_COUNTS,
            (157, "ipv4_relay_retry_14"): RELAY_157_WAVE14_ARCHIVED_SSE_COUNTS,
        }.get((seed, segment))
        if archived_failure_sse_counts is not None:
            reviewed_previous_api["archived_failure_sse_counts"] = archived_failure_sse_counts
    if (seed, segment) == (887, "ipv4_relay_retry_11") and latest == recovery_round:
        observation = RELAY_887_WAVE11_OBSERVATION
        reviewed_previous_api = {name: observation[name] for name in (
            "http_status", "error_type", "provider_error_code", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(
            complete_event_seen=False,
            output_text_done_seconds=observation["sse_milestones"]["response.output_text.done"]["elapsed_seconds"],
        )
    if (seed, segment) == (887, "ipv4_relay_retry_05") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_887_WAVE05_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if (seed, segment) == (887, "ipv4_relay_retry_06") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_887_WAVE06_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, hard_deadline_seconds=3600)
    if (seed, segment) == (887, "ipv4_relay_retry_08") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_887_WAVE08_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if (seed, segment) == (887, "ipv4_relay_retry_07") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_887_WAVE07_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, hard_deadline_seconds=3600,
                                     output_text_done_seconds=2861.863800625084)
    if (seed, segment) == (6101, "ipv4_relay_retry_04") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_6101_WAVE04_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if (seed, segment) == (6101, "ipv4_relay_retry_05") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_6101_WAVE05_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api["complete_event_seen"] = False
    if (seed, segment) == (6101, "ipv4_relay_retry_06") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_6101_WAVE06_OBSERVATION[name] for name in (
            "http_status", "error_type", "provider_error_code", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api["complete_event_seen"] = False
    if (seed, segment) == (6101, "ipv4_relay_retry_07") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_6101_WAVE07_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if (seed, segment) == (6101, "ipv4_relay_retry_08") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_6101_WAVE08_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, hard_deadline_seconds=3600)
    if (seed, segment) == (157, "ipv4_relay_retry_07") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_157_WAVE07_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    hard_wall = {(157, "ipv4_relay_retry_08"): RELAY_157_WAVE08_OBSERVATION,
                 (157, "ipv4_relay_retry_09"): RELAY_157_WAVE09_OBSERVATION,
                 (157, "ipv4_relay_retry_12"): RELAY_157_WAVE12_OBSERVATION,
                 (887, "ipv4_relay_retry_09"): RELAY_887_WAVE09_OBSERVATION}.get((seed, segment))
    if hard_wall is not None and latest == recovery_round:
        observation = hard_wall
        reviewed_previous_api = {name: observation[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, hard_deadline_seconds=3600)
    if (seed, segment) == (12203, "ipv4_relay_retry_08") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_12203_WAVE08_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count",
        )}
        reviewed_previous_api.update(complete_event_seen=False, event_type=RELAY_12203_WAVE08_OBSERVATION["conflicting_event_type"])
    if (seed, segment) == (12203, "ipv4_relay_retry_09") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_12203_WAVE09_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count",
        )}
        reviewed_previous_api.update(complete_event_seen=False, event_type=RELAY_12203_WAVE09_OBSERVATION["conflicting_event_type"])
    if (seed, segment) == (12203, "ipv4_relay_retry_10") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_12203_WAVE10_OBSERVATION[name] for name in (
            "http_status", "error_type", "event_type", "sse_event_count", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if seed == 12203 and segment in ("ipv4_relay_retry_11", "ipv4_relay_retry_12") and latest == recovery_round:
        observation = RELAY_12203_WAVE12_OBSERVATION if segment.endswith("_12") else RELAY_12203_WAVE11_OBSERVATION
        reviewed_previous_api = {name: observation[name] for name in (
            "http_status", "error_type", "sse_event_count", "response_id_matches_transport_alias",
        )}
        reviewed_previous_api.update(complete_event_seen=False, event_type=observation["conflicting_event_type"])
        if segment.endswith("_12"):
            reviewed_previous_api["archived_failure_count"] = 2
    if seed == 12203 and segment in ("ipv4_relay_retry_13", "ipv4_relay_retry_14") and latest == recovery_round:
        latest_wave = segment.endswith("_14")
        posted = RELAY_12203_WAVE14_OBSERVATION if latest_wave else RELAY_12203_WAVE12_POSTED_OBSERVATION
        reviewed_previous_api = {name: posted[name] for name in (
            "http_status", "error_type", "sse_event_count", "response_id_matches_transport_alias", "provider_error_code",
        )}
        reviewed_previous_api.update(complete_event_seen=False, event_type=posted["conflicting_event_type"])
        pre_post_failure = {"error_type": RELAY_12203_WAVE13_OBSERVATION["error_type"],
                            "network_phase": "tls", "model_post_observed": False}
        if latest_wave:
            reviewed_previous_api["earlier_pre_post_failure"] = {**pre_post_failure, "wave": 12}
        else:
            reviewed_previous_api["pre_post_failure"] = pre_post_failure
    if seed == 12203 and segment in ("ipv4_relay_retry_15", "ipv4_relay_retry_16") and latest == recovery_round:
        observation = RELAY_12203_WAVE16_OBSERVATION if segment.endswith("_16") else RELAY_12203_WAVE15_OBSERVATION
        reviewed_previous_api = {name: observation[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    inactivity_failure = {(157, "ipv4_relay_retry_10"): RELAY_157_WAVE10_OBSERVATION,
                          (157, "ipv4_relay_retry_13"): RELAY_157_WAVE13_OBSERVATION,
                          (6101, "ipv4_relay_retry_12"): RELAY_6101_WAVE12_OBSERVATION,
                          (6101, "ipv4_relay_retry_18"): RELAY_6101_WAVE18_OBSERVATION,
                          (887, "ipv4_relay_retry_10"): RELAY_887_WAVE10_OBSERVATION}.get((seed, segment))
    if inactivity_failure is not None and latest == recovery_round:
        reviewed_previous_api = {name: inactivity_failure[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
        if (seed, segment) == (157, "ipv4_relay_retry_10"):
            reviewed_previous_api["earlier_response_error_code"] = "upstream_error"
        if (seed, segment) == (157, "ipv4_relay_retry_13"):
            reviewed_previous_api.update(earlier_response_error_code=RELAY_157_WAVE13_EARLIER_ERROR,
                                         earlier_response_sse_count=RELAY_157_WAVE13_ARCHIVED_SSE_COUNTS[0])
    if (seed, segment) == (12203, "ipv4_relay_retry_18") and latest == recovery_round:
        observation = RELAY_12203_WAVE18_OBSERVATION
        reviewed_previous_api = {name: observation[name] for name in (
            "http_status", "error_type", "sse_event_count", "provider_error_code", "response_id_matches_transport_alias",
        )}
        reviewed_previous_api.update(complete_event_seen=False, event_type=observation["conflicting_event_type"])
    if (seed, segment) == (12203, "ipv4_relay_retry_19") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_12203_WAVE19_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900,
                                     earlier_response_error_code="upstream_error",
                                     earlier_response_sse_count=RELAY_12203_WAVE19_ARCHIVED_SSE_COUNTS[0])
    without_headers = {(157, "ipv4_relay_retry_11"): RELAY_157_WAVE11_OBSERVATION,
                       (2357, "ipv4_relay_retry_10"): RELAY_2357_WAVE10_OBSERVATION,
                       (2357, "ipv4_relay_retry_19"): RELAY_2357_WAVE19_OBSERVATION,
                       (12203, "ipv4_relay_retry_17"): RELAY_12203_WAVE17_OBSERVATION,
                       (6101, "ipv4_relay_retry_09"): RELAY_6101_WAVE09_OBSERVATION}.get((seed, segment))
    if latest == recovery_round and without_headers is not None:
        phases = without_headers.get("network_phases", {})
        upload_state = phases.get("send_body", {}).get("state")
        reviewed_previous_api = {"http_status": None, "sse_event_count": 0, "error_type": without_headers["error_type"],
                                 "complete_event_seen": False, "no_http_headers": True,
                                 "request_upload_confirmed": {"complete": True, "failed": False}.get(upload_state)}
        if (seed, segment) == (2357, "ipv4_relay_retry_19"):
            reviewed_previous_api.update(earlier_response_error_code="upstream_error",
                                         earlier_response_sse_count=RELAY_2357_WAVE19_ARCHIVED_SSE_COUNTS[0],
                                         receive_headers_seconds=phases["receive_headers"]["duration_seconds"])
    boundary_api = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0} if local_fit else {}
    if seed == 2357 and segment == "ipv4_relay_retry_03" and latest == recovery_round:
        boundary_api = {"maximum_logical_requests": 1,
                        "maximum_physical_attempts": RELAY_2357_WAVE03_PROOF["newly_authorized_boundary_physical_attempts"],
                        "replay_delay_seconds": RELAY_RETRY_SECONDS}
    counted_boundary = {
        (6101, "ipv4_relay_retry_04"): RELAY_6101_WAVE04_PROOF,
        (6101, "ipv4_relay_retry_05"): RELAY_6101_WAVE05_PROOF,
        (6101, "ipv4_relay_retry_06"): RELAY_6101_WAVE06_PROOF,
        (6101, "ipv4_relay_retry_07"): RELAY_6101_WAVE07_PROOF,
        (6101, "ipv4_relay_retry_08"): RELAY_6101_WAVE08_PROOF,
        (6101, "ipv4_relay_retry_09"): RELAY_6101_WAVE09_PROOF,
        (6101, "ipv4_relay_retry_10"): RELAY_6101_WAVE10_PROOF,
        (6101, "ipv4_relay_retry_11"): RELAY_6101_WAVE11_PROOF,
        (6101, "ipv4_relay_retry_12"): RELAY_6101_WAVE12_PROOF,
        (6101, "ipv4_relay_retry_13"): RELAY_6101_WAVE13_PROOF,
        (6101, "ipv4_relay_retry_14"): RELAY_6101_WAVE14_PROOF,
        (6101, "ipv4_relay_retry_15"): RELAY_6101_WAVE15_PROOF,
        (6101, "ipv4_relay_retry_16"): RELAY_6101_WAVE16_PROOF,
        (6101, "ipv4_relay_retry_17"): RELAY_6101_WAVE17_PROOF,
        (6101, "ipv4_relay_retry_18"): RELAY_6101_WAVE18_PROOF,
        (6101, "ipv4_relay_retry_19"): RELAY_6101_WAVE19_PROOF,
        (6101, "ipv4_relay_retry_20"): RELAY_6101_WAVE20_PROOF,
        (6101, "ipv4_relay_retry_21"): RELAY_6101_WAVE21_PROOF,
        (6101, "ipv4_relay_retry_22"): RELAY_6101_WAVE22_PROOF,
        (6101, "ipv4_relay_retry_23"): RELAY_6101_WAVE23_PROOF,
        (6101, "ipv4_relay_retry_24"): RELAY_6101_WAVE24_PROOF,
        (6101, "ipv4_relay_retry_25"): RELAY_6101_WAVE25_PROOF,
        (6101, "ipv4_relay_retry_26"): RELAY_6101_WAVE26_PROOF,
        (6101, "ipv4_relay_retry_27"): RELAY_6101_WAVE27_PROOF,
        (6101, "ipv4_relay_retry_28"): RELAY_6101_WAVE28_PROOF,
        (2357, "ipv4_relay_retry_04"): RELAY_2357_WAVE04_PROOF,
        (2357, "ipv4_relay_retry_05"): RELAY_2357_WAVE05_PROOF,
        (2357, "ipv4_relay_retry_06"): RELAY_2357_WAVE06_PROOF,
        (2357, "ipv4_relay_retry_07"): RELAY_2357_WAVE07_PROOF,
        (2357, "ipv4_relay_retry_08"): RELAY_2357_WAVE08_PROOF,
        (2357, "ipv4_relay_retry_09"): RELAY_2357_WAVE09_PROOF,
        (2357, "ipv4_relay_retry_10"): RELAY_2357_WAVE10_PROOF,
        (2357, "ipv4_relay_retry_11"): RELAY_2357_WAVE11_PROOF,
        (2357, "ipv4_relay_retry_12"): RELAY_2357_WAVE12_PROOF,
        (2357, "ipv4_relay_retry_13"): RELAY_2357_WAVE13_PROOF,
        (2357, "ipv4_relay_retry_14"): RELAY_2357_WAVE14_PROOF,
        (2357, "ipv4_relay_retry_15"): RELAY_2357_WAVE15_PROOF,
        (2357, "ipv4_relay_retry_16"): RELAY_2357_WAVE16_PROOF,
        (2357, "ipv4_relay_retry_17"): RELAY_2357_WAVE17_PROOF,
        (2357, "ipv4_relay_retry_18"): RELAY_2357_WAVE18_PROOF,
        (2357, "ipv4_relay_retry_19"): RELAY_2357_WAVE19_PROOF,
        (2357, "ipv4_relay_retry_20"): RELAY_2357_WAVE20_PROOF,
        (2357, "ipv4_relay_retry_21"): RELAY_2357_WAVE21_PROOF,
        (2357, "ipv4_relay_retry_22"): RELAY_2357_WAVE22_PROOF,
        (2357, "ipv4_relay_retry_23"): RELAY_2357_WAVE23_PROOF,
        (2357, "ipv4_relay_retry_24"): RELAY_2357_WAVE24_PROOF,
        (2357, "ipv4_relay_retry_25"): RELAY_2357_WAVE25_PROOF,
        (2357, "ipv4_relay_retry_26"): RELAY_2357_WAVE26_PROOF,
        (2357, "ipv4_relay_retry_27"): RELAY_2357_WAVE27_PROOF,
        (2357, "ipv4_relay_retry_28"): RELAY_2357_WAVE28_PROOF,
        (2357, "ipv4_relay_retry_29"): RELAY_2357_WAVE29_PROOF,
        (2357, "ipv4_relay_retry_30"): RELAY_2357_WAVE30_PROOF,
        (2357, "ipv4_relay_retry_31"): RELAY_2357_WAVE31_PROOF,
        (887, "ipv4_relay_retry_05"): RELAY_887_WAVE05_PROOF,
        (887, "ipv4_relay_retry_06"): RELAY_887_WAVE06_PROOF,
        (887, "ipv4_relay_retry_07"): RELAY_887_WAVE07_PROOF,
        (887, "ipv4_relay_retry_08"): RELAY_887_WAVE08_PROOF,
        (887, "ipv4_relay_retry_09"): RELAY_887_WAVE09_PROOF,
        (887, "ipv4_relay_retry_10"): RELAY_887_WAVE10_PROOF,
        (887, "ipv4_relay_retry_11"): RELAY_887_WAVE11_PROOF,
        (887, "ipv4_relay_retry_12"): RELAY_887_WAVE12_PROOF,
        (887, "ipv4_relay_retry_13"): RELAY_887_WAVE13_PROOF,
        (887, "ipv4_relay_retry_14"): RELAY_887_WAVE14_PROOF,
        (887, "ipv4_relay_retry_15"): RELAY_887_WAVE15_PROOF,
        (157, "ipv4_relay_retry_07"): RELAY_157_WAVE07_PROOF,
        (157, "ipv4_relay_retry_08"): RELAY_157_WAVE08_PROOF,
        (157, "ipv4_relay_retry_09"): RELAY_157_WAVE09_PROOF,
        (157, "ipv4_relay_retry_10"): RELAY_157_WAVE10_PROOF,
        (157, "ipv4_relay_retry_11"): RELAY_157_WAVE11_PROOF,
        (157, "ipv4_relay_retry_12"): RELAY_157_WAVE12_PROOF,
        (157, "ipv4_relay_retry_13"): RELAY_157_WAVE13_PROOF,
        (157, "ipv4_relay_retry_14"): RELAY_157_WAVE14_PROOF,
        (157, "ipv4_relay_retry_15"): RELAY_157_WAVE15_PROOF,
        (157, "ipv4_relay_retry_16"): RELAY_157_WAVE16_PROOF,
        (157, "ipv4_relay_retry_17"): RELAY_157_WAVE17_PROOF,
        (157, "ipv4_relay_retry_18"): RELAY_157_WAVE18_PROOF,
        (157, "ipv4_relay_retry_19"): RELAY_157_WAVE19_PROOF,
        (157, "ipv4_relay_retry_20"): RELAY_157_WAVE20_PROOF,
        (157, "ipv4_relay_retry_21"): RELAY_157_WAVE21_PROOF,
        (157, "ipv4_relay_retry_22"): RELAY_157_WAVE22_PROOF,
        (157, "ipv4_relay_retry_23"): RELAY_157_WAVE23_PROOF,
        (157, "ipv4_relay_retry_24"): RELAY_157_WAVE24_PROOF,
        (157, "ipv4_relay_retry_25"): RELAY_157_WAVE25_PROOF,
        (157, "ipv4_relay_retry_26"): RELAY_157_WAVE26_PROOF,
        (157, "ipv4_relay_retry_27"): RELAY_157_WAVE27_PROOF,
        (157, "ipv4_relay_retry_28"): RELAY_157_WAVE28_PROOF,
        (157, "ipv4_relay_retry_29"): RELAY_157_WAVE29_PROOF,
        (157, "ipv4_relay_retry_30"): RELAY_157_WAVE30_PROOF,
        (157, "ipv4_relay_retry_31"): RELAY_157_WAVE31_PROOF,
        (12203, "ipv4_relay_retry_07"): RELAY_12203_WAVE07_PROOF,
        (12203, "ipv4_relay_retry_08"): RELAY_12203_WAVE08_PROOF,
        (12203, "ipv4_relay_retry_09"): RELAY_12203_WAVE09_PROOF,
        (12203, "ipv4_relay_retry_10"): RELAY_12203_WAVE10_PROOF,
        (12203, "ipv4_relay_retry_11"): RELAY_12203_WAVE11_PROOF,
        (12203, "ipv4_relay_retry_12"): RELAY_12203_WAVE12_PROOF,
        (12203, "ipv4_relay_retry_13"): RELAY_12203_WAVE13_PROOF,
        (12203, "ipv4_relay_retry_14"): RELAY_12203_WAVE14_PROOF,
        (12203, "ipv4_relay_retry_15"): RELAY_12203_WAVE15_PROOF,
        (12203, "ipv4_relay_retry_16"): RELAY_12203_WAVE16_PROOF,
        (12203, "ipv4_relay_retry_17"): RELAY_12203_WAVE17_PROOF,
        (12203, "ipv4_relay_retry_18"): RELAY_12203_WAVE18_PROOF,
        (12203, "ipv4_relay_retry_19"): RELAY_12203_WAVE19_PROOF,
        (12203, "ipv4_relay_retry_20"): RELAY_12203_WAVE20_PROOF,
        (12203, "ipv4_relay_retry_21"): RELAY_12203_WAVE21_PROOF,
        (12203, "ipv4_relay_retry_22"): RELAY_12203_WAVE22_PROOF,
        (12203, "ipv4_relay_retry_23"): RELAY_12203_WAVE23_PROOF,
        (12203, "ipv4_relay_retry_24"): RELAY_12203_WAVE24_PROOF,
        (12203, "ipv4_relay_retry_25"): RELAY_12203_WAVE25_PROOF,
        (12203, "ipv4_relay_retry_26"): RELAY_12203_WAVE26_PROOF,
        (12203, "ipv4_relay_retry_27"): RELAY_12203_WAVE27_PROOF,
        (12203, "ipv4_relay_retry_28"): RELAY_12203_WAVE28_PROOF,
        (12203, "ipv4_relay_retry_29"): RELAY_12203_WAVE29_PROOF,
        (12203, "ipv4_relay_retry_30"): RELAY_12203_WAVE30_PROOF,
        (12203, "ipv4_relay_retry_31"): RELAY_12203_WAVE31_PROOF,
        (12203, "ipv4_relay_retry_32"): RELAY_12203_WAVE32_PROOF,
        (12203, "ipv4_relay_retry_33"): RELAY_12203_WAVE33_PROOF,
        (12203, "ipv4_relay_retry_34"): RELAY_12203_WAVE34_PROOF,
        (12203, "ipv4_relay_retry_35"): RELAY_12203_WAVE35_PROOF,
        (12203, "ipv4_relay_retry_36"): RELAY_12203_WAVE36_PROOF,
        (12203, "ipv4_relay_retry_37"): RELAY_12203_WAVE37_PROOF,
    }.get((seed, segment))
    if dynamic_relay is not None:
        counted_boundary = dynamic_relay
        observation = dynamic_relay["previous_transport_observation"]
        reviewed_previous_api = {key: observation[key] for key in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        ) if key in observation}
        reviewed_previous_api.update(complete_event_seen=False, archived_failure_count=len(dynamic_relay["previous_attempts"]))
    if counted_boundary is not None and latest == recovery_round:
        maximum = counted_boundary["newly_authorized_boundary_physical_attempts"]
        boundary_api = {"maximum_logical_requests": 1, "maximum_physical_attempts": maximum,
                        "replay_delay_seconds": RELAY_RETRY_SECONDS,
                        **boundary_attempt_counts(run, directory, latest, maximum)}
        if (seed, segment) == (2357, "ipv4_relay_retry_05"):
            boundary_api["future_replay_from_round"] = 5
        if (seed, segment) == (2357, "ipv4_relay_retry_06"):
            boundary_api.update(future_replay_from_round=5, future_replay_requires_verified_response_id=True,
                                replay_requires_verified_response_id=True)
        if (seed, segment) == (2357, "ipv4_relay_retry_07"):
            boundary_api.update(future_replay_from_round=5, future_replay_requires_verified_response_id=True)
        if seed == 887 and segment in ("ipv4_relay_retry_05", "ipv4_relay_retry_07", "ipv4_relay_retry_09"):
            boundary_api.update(future_replay_from_round=13, future_replay_requires_verified_response_id=True)
        if seed == 887 and segment in ("ipv4_relay_retry_06", "ipv4_relay_retry_08"):
            boundary_api.update(future_replay_from_round=13, future_replay_requires_verified_response_id=True,
                                replay_requires_verified_response_id=True)
        if (seed, segment) in ((157, "ipv4_relay_retry_11"), (2357, "ipv4_relay_retry_10"),
                               (6101, "ipv4_relay_retry_09"), (6101, "ipv4_relay_retry_12"), (6101, "ipv4_relay_retry_18"),
                               (887, "ipv4_relay_retry_10"), (887, "ipv4_relay_retry_11")):
            boundary_api.update(future_replay_from_round=recovery_round + 1, future_replay_requires_verified_response_id=True)
        if seed == 6101 and segment in ("ipv4_relay_retry_07", "ipv4_relay_retry_08"):
            boundary_api.update(future_replay_from_round=4, future_replay_requires_verified_response_id=True)
            if segment.endswith("_08"):
                boundary_api["replay_requires_verified_response_id"] = True
        if (seed, segment) == (12203, "ipv4_relay_retry_10"):
            boundary_api.update(future_replay_from_round=6, future_replay_requires_verified_response_id=True)
        if seed == 12203 and segment in ("ipv4_relay_retry_13", "ipv4_relay_retry_15", "ipv4_relay_retry_17"):
            boundary_api.update(future_replay_from_round=6, future_replay_requires_verified_response_id=True)
        if seed == 12203 and segment in ("ipv4_relay_retry_11", "ipv4_relay_retry_12", "ipv4_relay_retry_14",
                                        "ipv4_relay_retry_16", "ipv4_relay_retry_18", "ipv4_relay_retry_19",
                                        "ipv4_relay_retry_20", "ipv4_relay_retry_21", "ipv4_relay_retry_22", "ipv4_relay_retry_23",
                                        "ipv4_relay_retry_24", "ipv4_relay_retry_25", "ipv4_relay_retry_26",
                                        "ipv4_relay_retry_27", "ipv4_relay_retry_28", "ipv4_relay_retry_29",
                                        "ipv4_relay_retry_30", "ipv4_relay_retry_31", "ipv4_relay_retry_32",
                                        "ipv4_relay_retry_33", "ipv4_relay_retry_34", "ipv4_relay_retry_35", "ipv4_relay_retry_36", "ipv4_relay_retry_37"):
            boundary_api.update(future_replay_from_round=6, future_replay_requires_verified_response_id=True,
                                replay_requires_verified_response_id=True)
        if (seed, segment) in ((157, "ipv4_relay_retry_08"), (157, "ipv4_relay_retry_09"), (157, "ipv4_relay_retry_10"),
                               (157, "ipv4_relay_retry_12"), (157, "ipv4_relay_retry_13"),
                               (2357, "ipv4_relay_retry_08"), (2357, "ipv4_relay_retry_09"),
                               (2357, "ipv4_relay_retry_11"), (2357, "ipv4_relay_retry_12"),
                               (2357, "ipv4_relay_retry_13"), (2357, "ipv4_relay_retry_14"),
                               (157, "ipv4_relay_retry_14"), (2357, "ipv4_relay_retry_15"),
                               (6101, "ipv4_relay_retry_10"), (6101, "ipv4_relay_retry_11"),
                               (157, "ipv4_relay_retry_15"), (157, "ipv4_relay_retry_16"), (2357, "ipv4_relay_retry_16"),
                               (2357, "ipv4_relay_retry_17"), (6101, "ipv4_relay_retry_13"), (6101, "ipv4_relay_retry_14"),
                               (6101, "ipv4_relay_retry_15"), (6101, "ipv4_relay_retry_16"),
                               (157, "ipv4_relay_retry_17"), (157, "ipv4_relay_retry_18"), (157, "ipv4_relay_retry_19"),
                               (2357, "ipv4_relay_retry_18"), (2357, "ipv4_relay_retry_19"),
                               (6101, "ipv4_relay_retry_17"), (6101, "ipv4_relay_retry_19"),
                               (157, "ipv4_relay_retry_20"), (2357, "ipv4_relay_retry_20"),
                               (6101, "ipv4_relay_retry_20"), (157, "ipv4_relay_retry_21"),
                               (2357, "ipv4_relay_retry_21"), (6101, "ipv4_relay_retry_21"),
                               (157, "ipv4_relay_retry_22"), (2357, "ipv4_relay_retry_22"),
                               (2357, "ipv4_relay_retry_23"),
                               (2357, "ipv4_relay_retry_24"),
                               (2357, "ipv4_relay_retry_25"),
                               (2357, "ipv4_relay_retry_26"),
                               (2357, "ipv4_relay_retry_27"),
                               (2357, "ipv4_relay_retry_28"),
                               (2357, "ipv4_relay_retry_29"),
                               (2357, "ipv4_relay_retry_30"),
                               (2357, "ipv4_relay_retry_31"),
                               (157, "ipv4_relay_retry_23"),
                               (157, "ipv4_relay_retry_24"),
                               (157, "ipv4_relay_retry_25"),
                               (157, "ipv4_relay_retry_26"),
                               (157, "ipv4_relay_retry_27"),
                               (157, "ipv4_relay_retry_28"),
                               (157, "ipv4_relay_retry_29"),
                               (157, "ipv4_relay_retry_30"),
                               (157, "ipv4_relay_retry_31"),
                               (6101, "ipv4_relay_retry_23"),
                               (6101, "ipv4_relay_retry_24"),
                               (6101, "ipv4_relay_retry_25"),
                               (6101, "ipv4_relay_retry_26"),
                               (6101, "ipv4_relay_retry_27"),
                               (6101, "ipv4_relay_retry_28"),
                               (887, "ipv4_relay_retry_12"),
                               (887, "ipv4_relay_retry_13"),
                               (887, "ipv4_relay_retry_14"),
                               (887, "ipv4_relay_retry_15"),
                               (6101, "ipv4_relay_retry_22")):
            boundary_api.update(future_replay_from_round=recovery_round + 1, future_replay_requires_verified_response_id=True,
                                replay_requires_verified_response_id=True)
    if dynamic_relay is not None and boundary_api:
        boundary_api.update(future_replay_from_round=recovery_round + 1, future_replay_requires_verified_response_id=True,
                            replay_requires_verified_response_id=boundary_api["maximum_physical_attempts"] == 2)
    if segment in ("ipv4_6101_r8_targeted_repair", "ipv4_887_r13_targeted_repair",
                   "ipv4_887_r32_targeted_repair") and latest == recovery_round:
        boundary_api = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1,
                        **boundary_attempt_counts(run, directory, latest, 1)}
    if segment in ("ipv4_887_r9_continuation", "ipv4_6101_r5_continuation",
                   "ipv4_887_r14_continuation", "ipv4_887_r20_continuation") and latest == recovery_round:
        boundary_api = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2,
                        "replay_delay_seconds": RELAY_RETRY_SECONDS,
                        **boundary_attempt_counts(run, directory, latest, 2)}
        if segment in ("ipv4_6101_r5_continuation", "ipv4_887_r14_continuation", "ipv4_887_r20_continuation"):
            boundary_api.update(replay_requires_verified_response_id=True,
                                future_replay_from_round=recovery_round + 1, future_replay_requires_verified_response_id=True)
    if segment in ("ipv4_887_r19_cached", "ipv4_887_r20_continuation"):
        cache_import = {"round": 19, "fitted_labels": 973, "choice_count": 32, "removed_reference_fields": 0,
                        "provider_dispatched": False, "archived_failed_correction": "invalid_response_message_phase"}
    if segment in ("ipv4_887_r13_cached_correction", "ipv4_887_r14_continuation"):
        cache_import = {"round": 13, "fitted_labels": 781, "choice_count": 32, "removed_reference_fields": 61,
                        "provider_dispatched": False}
    if seed == 157 and segment in ("ipv4_cached_r3", "ipv4_relay_retry_07", "ipv4_relay_retry_08", "ipv4_relay_retry_09",
                                   "ipv4_relay_retry_10", "ipv4_relay_retry_11", "ipv4_relay_retry_12",
                                   "ipv4_relay_retry_13", "ipv4_relay_retry_14", "ipv4_relay_retry_15",
                                   "ipv4_relay_retry_16", "ipv4_relay_retry_17", "ipv4_relay_retry_18", "ipv4_relay_retry_19",
                                   "ipv4_relay_retry_20", "ipv4_relay_retry_21", "ipv4_relay_retry_22",
                                   "ipv4_relay_retry_23", "ipv4_relay_retry_24", "ipv4_relay_retry_25",
                                   "ipv4_relay_retry_26", "ipv4_relay_retry_27", "ipv4_relay_retry_28", "ipv4_relay_retry_29", "ipv4_relay_retry_30", "ipv4_relay_retry_31"):
        cache_import = {"round": 3, "fitted_labels": 461, "choice_count": 32, "removed_reference_fields": 79,
                        "provider_dispatched": False}
    if latest == TARGET_ROUNDS:
        phase = "complete"
    elif (local_fit and latest == 5) or segment == "ipv4_6101_r10_cached":
        phase = "cached_import_complete"
    elif local_fit and not paused and phase != "failed" and gpu.get("stage") not in ("waiting", "cuda"):
        phase = "validating_cached_response" if worker_live else "cached_fit_unverified"
    elif ((segment == "ipv4_r8_cached_correction" and latest == 9) or (local_887_fit and latest == 14)
          or (local_887_r19_fit and latest == 20)):
        phase = "cached_correction_round_complete"
    elif segment in ("ipv4_r8_targeted_repair", "ipv4_887_r13_targeted_repair",
                     "ipv4_887_r32_targeted_repair", "ipv4_6101_r8_targeted_repair") and latest == recovery_round + 1:
        phase = "targeted_repair_round_complete"
    elif segment in ("ipv4_extension", "ipv4_json_extension", "ipv4_ws_extension") and latest == IPV4_EXTENSION_END_ROUNDS[seed] and (
        run / "extension_completion.json"
    ).is_file():
        phase = "transport_extension_complete"
    elif segment == "ipv4_bounded_sse" and latest == BOUNDED_SSE_END_ROUNDS[seed] and (run / "extension_completion.json").is_file():
        phase = "transport_extension_complete"
    elif segment == "ipv4_r2_extension" and latest == 3 and (run / "extension_completion.json").is_file():
        phase = "transport_extension_complete"
    elif latest == PILOT_ROUNDS and segment not in ("v6", "v11", "background", "ipv4", "ipv4_continuation",
                                                     "ipv4_r2_continuation", *RESILIENT_SEGMENTS):
        phase = "awaiting_v6_handoff"
    elif (cache_import and segment != "ipv4_6101_r5_continuation" and latest == cache_import["round"] + 1
          and segment not in ("ipv4_887_r14_continuation", "ipv4_887_r20_continuation")
          and not worker_live and not api and not paused):
        phase = "cached_import_complete"
    elif paused:
        phase = "paused"
    elif latest == 5 and segment == "repair":
        phase = "targeted_repair_round_complete"
    elif latest == (5 if seed == 887 else 2) and segment == "direct":
        phase = "direct_round_complete"
    elif phase == "failed" and api.get("selection") in {"rejected", "malformed_complete_response", "patch_rejected"}:
        phase = "selection_rejected"
    elif phase in {"paused", "failed", "bridge_pending"}:
        pass
    elif worker_live and gpu.get("stage") in ("waiting", "cuda"):
        phase = f"gpu_{gpu['stage']}"
    elif worker_live and "relay_retry_wait_seconds" in api:
        phase = "relay_retry_cooldown" if api["relay_retry_wait_seconds"] else "awaiting_relay_retry"
    elif api.get("status") in {"waiting_for_mac", "returned_from_mac"}:
        phase = api["status"]
    elif worker_live and api.get("status") == "completed":
        phase = "selection_or_repair"
    elif not worker_live and phase == "initializing":
        phase = "worker_stopped"
    return {"seed": seed, "run": run.name, "contract_sha256": contract, "segment": segment,
            "continuation": segment in ("ipv4", "ipv4_extension", "ipv4_continuation", "ipv4_r2_extension",
                                        "ipv4_r2_continuation", "ipv4_json_extension", "ipv4_ws_extension",
                                        "ipv4_bounded_sse", *RESILIENT_SEGMENTS, "background", "v11", "v6", "v5",
                                        "direct", "repair"),
            **({"recovery_round": recovery_round, "readiness_windows": readiness_windows}
               if segment in RESILIENT_SEGMENTS else {}),
            **({"sealed_relay_retry_seconds": relay_retry} if relay_retry is not None else {}),
            **({"relay_not_before_unix": relay_boundary_not_before} if relay_boundary_not_before is not None else {}),
            **({"reviewed_previous_api": reviewed_previous_api} if reviewed_previous_api else {}),
            **({"response_headers_seconds": 300}
               if seed == 2357 and (dynamic_relay is not None or segment in ("ipv4_relay_retry_19", "ipv4_relay_retry_20", "ipv4_relay_retry_21",
                                               "ipv4_relay_retry_22", "ipv4_relay_retry_23", "ipv4_relay_retry_24",
                                               "ipv4_relay_retry_25", "ipv4_relay_retry_26", "ipv4_relay_retry_27", "ipv4_relay_retry_28", "ipv4_relay_retry_29", "ipv4_relay_retry_30", "ipv4_relay_retry_31"))
               else {}),
            **({"boundary_api": boundary_api} if boundary_api else {}),
            **({"cached_import": cache_import} if cache_import else {}),
            **({"archived_correction_provider_post": False} if seed == 887 and segment in (
                "ipv4_r8_cached_correction", "ipv4_887_r9_continuation", "ipv4_relay_retry_05", "ipv4_relay_retry_06",
                "ipv4_relay_retry_07", "ipv4_relay_retry_08", "ipv4_relay_retry_09", "ipv4_relay_retry_10",
                "ipv4_887_r13_cached_correction", "ipv4_887_r14_continuation",
                "ipv4_887_r19_cached", "ipv4_887_r20_continuation") else {}),
            **({"segment_target_labels": 1389} if segment == "ipv4_887_r32_targeted_repair" else {}),
            **({"segment_target_labels": 621} if segment == "ipv4_6101_r8_targeted_repair" else {}),
            **({"targeted_repair": {"round": 8, "fitted_labels": 621, "choice_count": 32, "repaired_references": 2},
                "normal_api_limits": {"logical_per_round": 3, "physical_per_logical": 2, "physical_total": 162}}
               if segment == "ipv4_6101_r9_continuation" else {}),
            **({"targeted_repair": {"round": 8, "fitted_labels": 621, "choice_count": 32, "repaired_references": 2}}
               if segment in ("ipv4_6101_r10_cached", "ipv4_6101_r11_continuation") else {}),
            **({"normal_api_limits": {"logical_per_round": 3, "physical_per_logical": 2, "physical_total": 150}}
               if segment == "ipv4_6101_r11_continuation" else {}),
            **({"normal_api_limits": {"logical_per_round": 3, "physical_per_logical": 2,
                                       "physical_total": 18}} if segment == "ipv4_887_r33_continuation" else {}),
            **({"remaining_api_limits": {"round": 35, "logical": 3, "physical_per_logical": 2, "physical_total": 8}}
               if seed == 887 and (dynamic_relay is not None or segment in (
                   "ipv4_relay_retry_12", "ipv4_relay_retry_13", "ipv4_relay_retry_14", "ipv4_relay_retry_15"))
               else {}),
            "target_rounds": (5 if local_fit else 20 if local_887_r19_fit else
                              14 if local_887_fit or segment == "ipv4_887_r13_targeted_repair" else TARGET_ROUNDS),
            "target_labels": (493 if local_fit else 973 if local_887_r19_fit else
                              781 if local_887_fit or segment == "ipv4_887_r13_targeted_repair" else TARGET_LABELS),
            "completed_rounds": latest, "labels": 333 + latest * 32 if latest is not None else None,
            "phase": phase, "worker_pid": worker_pid, "worker_active": worker_live,
            "worker_error": safe_code(worker.get("error_type")), "coordinator_pid": coordinator,
            "coordinator_active": coordinator_live, "gpu": gpu.get("gpu"), "api": api, **metrics}


def command_lines(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=8).splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def remote_report(runs=RUNS, relay_pins=None):
    return {"seeds": [remote_seed(runs, seed, relay_pins) if relay_pins else remote_seed(runs, seed) for seed in SEEDS], "gpus": command_lines([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ]), "gpu_processes": command_lines([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits",
    ])}


def usage(data):
    return {key: data[key] for key in TOKEN_FIELDS if isinstance(data, dict) and type(data.get(key)) is int}


def seconds(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def mac_start_time(marker):
    try:
        parsed = datetime.fromisoformat(marker["started_utc"])
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None


def route_readiness(spool, attempt, identity, seed, active, windows):
    """Read only receipts for the exact current reservation, before any model start."""
    if windows != 3:
        return {}
    job = read_json(spool / "jobs" / f"{identity}.json")
    job_sha = job.get("job_sha256")
    request = job.get("request") or {}
    binding = request.get("binding") or {}
    if ((job.get("attempt_id"), binding.get("seed"), binding.get("round"), request.get("logical_index"))
            != (identity, seed, active.get("round"), active.get("turn"))
            or not isinstance(job_sha, str) or not SHA.fullmatch(job_sha)):
        return {}
    files = sorted(attempt.glob("readiness_[0-9][0-9].json"))
    if not files or len(files) > windows:
        return {}
    for index, path in enumerate(files, 1):
        latest = read_json(path)
        if (path.name != f"readiness_{index:02d}.json"
                or (latest.get("attempt_id"), latest.get("job_sha256"), latest.get("window")) != (identity, job_sha, index)
                or seconds(latest.get("retry_unix")) is None):
            return {}
    ready = {"readiness_windows_used": len(files), "readiness_windows": windows}
    if (attempt / "started.json").exists():
        return ready
    result = read_json(spool / "results" / f"{identity}.json")
    ack = read_json(spool / "delivered" / f"{identity}.json")
    result_sha = result.get("result_sha256")
    if (result.get("attempt_id") == identity and result.get("job_sha256") == job_sha
            and result.get("provider_dispatched") is False and (result.get("outcome") or {}).get("status") == "failed"
            and isinstance(result_sha, str) and SHA.fullmatch(result_sha)):
        return {**ready, "phase": "transport_failure", "provider_dispatched": False, "sealed": True,
                "delivered": ack == {"attempt_id": identity, "result_sha256": result_sha}}
    retry = {"readiness_retry_seconds": max(0, round(latest["retry_unix"] - time.time()))} if len(files) < windows else {}
    return {**ready, **retry, "phase": "waiting_for_ipv4_route", "provider_dispatched": False}


def local_progress(spool, seed, active, *, track_queue=False, readiness_windows=0):
    """Report only the Mac attempt selected by the current GPFS ledger."""
    identity = active.get("attempt_id")
    if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
        return {}
    attempt = spool / "attempts" / identity
    path = attempt / "provider_progress.json"
    status = read_json(path)
    marker = read_json(attempt / "started.json")
    now = time.time()
    start = mac_start_time(marker) if marker.get("attempt_id") == identity else None
    readiness = route_readiness(spool, attempt, identity, seed, active, readiness_windows)
    timing = {}
    if track_queue and (reserved := seconds(active.get("reserved_unix"))) is not None:
        if start is not None:
            timing["dispatch_wait_seconds"] = seconds(start - reserved)
        elif not marker and active.get("status") == "waiting_for_mac":
            timing = {"phase": "queued_for_dispatch", "dispatch_wait_seconds": seconds(now - reserved)}
    if (status.get("seed") != seed or status.get("attempt_id") != identity or marker.get("attempt_id") != identity
            or not marker.get("job_sha256") or status.get("job_sha256") != marker["job_sha256"]
            or status.get("round") != active.get("round") or status.get("logical_index") != active.get("turn")
            or status.get("phase") not in PHASES):
        return {**timing, **readiness}
    try:
        updated = path.stat().st_mtime
    except OSError:
        return {**timing, **readiness}
    result = read_json(spool / "results" / f"{identity}.json")
    ack = read_json(spool / "delivered" / f"{identity}.json")
    sealed = result.get("attempt_id") == identity and result.get("job_sha256") == marker["job_sha256"]
    delivered = sealed and ack == {"attempt_id": identity, "result_sha256": result.get("result_sha256")}
    events = status.get("sse_milestones", {})
    if not isinstance(events, dict):
        events = {}
    first_text_event = events.get("response.output_text.delta")
    first_text = first_text_event.get("elapsed_seconds") if isinstance(first_text_event, dict) else None
    terminal_event = events.get("response.completed")
    terminal_time = terminal_event.get("elapsed_seconds") if isinstance(terminal_event, dict) else None
    tokens = {field: value for field in ("response_id", "conflicting_response_id", "provider_request_id",
                                         *PROVIDER_DIAGNOSTICS) if (value := provider_token(status.get(field))) is not None}
    diagnostics = {}
    provider_status = status.get("provider_error_status")
    if type(provider_status) is int and 400 <= provider_status <= 599:
        diagnostics["provider_error_status"] = provider_status
    alias = status.get("response_id_matches_transport_alias")
    if type(alias) is bool:
        diagnostics["response_id_matches_transport_alias"] = alias
    fields = ("round", "logical_index", "http_status", "elapsed_seconds", "sse_event_count")
    progress = {field: status[field] for field in fields if type(status.get(field)) in (int, float)}
    elapsed = seconds(status.get("elapsed_seconds"))
    if not sealed and status["phase"] not in ("transport_completed", "transport_failure"):
        if start is not None and (running := seconds(now - start)) is not None:
            elapsed = max(elapsed or 0, running)
    dispatch = {"provider_dispatched": result["provider_dispatched"]} if (
        sealed and type(result.get("provider_dispatched")) is bool
    ) else {}
    return {**timing, **readiness, **progress, **dispatch, "phase": status["phase"], "sealed": sealed, "delivered": delivered,
            "progress_age_seconds": max(0, round(now - updated)),
            "event_type": safe_code(status.get("event_type")), "usage": usage(status.get("usage")),
            **tokens, **diagnostics,
            "provider_elapsed_seconds": elapsed,
            "first_output_seconds": seconds(first_text),
            "last_sse_seconds": seconds(status.get("last_sse_seconds")),
            "last_sse_activity_seconds": seconds(status.get("last_sse_activity_seconds")),
            "last_sse_interval_seconds": seconds(status.get("last_sse_interval_seconds")),
            "complete_event_seen": (status.get("complete_terminal_captured") is True or "response.completed" in events
                                    or status.get("event_type") in {"response.completed", "background.completed"}),
            "complete_event_seconds": seconds(terminal_time),
            "raw_response_saved": (attempt / "raw_provider_response.json").is_file()
                                  and (attempt / "provider_capture_receipt.json").is_file()}


def dispatcher_pid(seed, segment):
    relay = re.fullmatch(r"ipv4_relay_retry_([0-9]{2})", segment)
    if relay and seed in SEEDS:
        label = f"gui/{os.getuid()}/com.purellm.a7-ipv4-v13-{seed}-relay-retry-{relay[1]}"
        return launchd_service_pid(label)
    if segment in ("ipv4_r8_cached_correction", "ipv4_cached_r4_6101"):
        return None
    cohort_services = {"background": "a7-background-v12", "ipv4": "a7-ipv4-v13",
                       "ipv4_extension": f"a7-ipv4-v13-{seed}-extension-01",
                       "ipv4_continuation": f"a7-ipv4-v13-{seed}-continuation-01",
                       "ipv4_r2_extension": "a7-ipv4-v13-2357-r2-extension-01",
                       "ipv4_r2_continuation": "a7-ipv4-v13-2357-r2-continuation-01",
                       "ipv4_json_extension": "a7-ipv4-v13-887-json-extension-01",
                       "ipv4_ws_extension": "a7-ipv4-v13-6101-ws-extension-01",
                       "ipv4_bounded_sse": f"a7-ipv4-v13-{seed}-bounded-sse-01",
                       "ipv4_resilient_sse": f"a7-ipv4-v13-{seed}-resilient-sse-01",
                       "ipv4_resilient_failure": f"a7-ipv4-v13-{seed}-resilient-failure-01",
                       "ipv4_relay_retry": f"a7-ipv4-v13-{seed}-relay-retry-01",
                       "ipv4_relay_retry_02": f"a7-ipv4-v13-{seed}-relay-retry-02",
                       "ipv4_relay_retry_03": f"a7-ipv4-v13-{seed}-relay-retry-03",
                       "ipv4_relay_retry_04": f"a7-ipv4-v13-{seed}-relay-retry-04",
                       "ipv4_relay_retry_05": f"a7-ipv4-v13-{seed}-relay-retry-05",
                       "ipv4_relay_retry_06": f"a7-ipv4-v13-{seed}-relay-retry-06",
                       "ipv4_r8_targeted_repair": "a7-ipv4-v13-887-r8-targeted-repair-01",
                       "ipv4_887_r13_targeted_repair": "a7-ipv4-v13-887-r13-targeted-repair-01",
                       "ipv4_887_r32_targeted_repair": "a7-ipv4-v13-887-r32-targeted-repair-01",
                       "ipv4_6101_r8_targeted_repair": "a7-ipv4-v13-6101-r8-targeted-repair-01",
                       "ipv4_6101_r9_continuation": "a7-ipv4-v13-6101-r9-continuation-01",
                       "ipv4_6101_r11_continuation": "a7-ipv4-v13-6101-r11-continuation-01",
                       "ipv4_887_r33_continuation": "a7-ipv4-v13-887-r33-continuation-01",
                       "ipv4_887_r14_continuation": "a7-ipv4-v13-887-r14-continuation-01",
                       "ipv4_887_r20_continuation": "a7-ipv4-v13-887-r20-continuation-01",
                       "ipv4_887_r9_continuation": "a7-ipv4-v13-887-r9-continuation-01",
                       "ipv4_6101_r5_continuation": "a7-ipv4-v13-6101-r5-continuation-01",
                       "ipv4_relay_retry_07": f"a7-ipv4-v13-{seed}-relay-retry-07",
                       "ipv4_relay_retry_08": f"a7-ipv4-v13-{seed}-relay-retry-08",
                       "ipv4_relay_retry_09": f"a7-ipv4-v13-{seed}-relay-retry-09",
                       "ipv4_relay_retry_10": f"a7-ipv4-v13-{seed}-relay-retry-10",
                       "ipv4_relay_retry_11": f"a7-ipv4-v13-{seed}-relay-retry-11",
                       "ipv4_relay_retry_12": f"a7-ipv4-v13-{seed}-relay-retry-12",
                       "ipv4_relay_retry_13": f"a7-ipv4-v13-{seed}-relay-retry-13",
                       "ipv4_relay_retry_14": f"a7-ipv4-v13-{seed}-relay-retry-14",
                       "ipv4_relay_retry_15": f"a7-ipv4-v13-{seed}-relay-retry-15",
                       "ipv4_relay_retry_16": f"a7-ipv4-v13-{seed}-relay-retry-16",
                       "ipv4_relay_retry_17": f"a7-ipv4-v13-{seed}-relay-retry-17",
                       "ipv4_relay_retry_18": f"a7-ipv4-v13-{seed}-relay-retry-18",
                       "ipv4_relay_retry_19": f"a7-ipv4-v13-{seed}-relay-retry-19",
                       "ipv4_relay_retry_20": f"a7-ipv4-v13-{seed}-relay-retry-20",
                       "ipv4_relay_retry_21": f"a7-ipv4-v13-{seed}-relay-retry-21",
                       "ipv4_relay_retry_22": f"a7-ipv4-v13-{seed}-relay-retry-22",
                       "ipv4_relay_retry_23": f"a7-ipv4-v13-{seed}-relay-retry-23",
                       "ipv4_relay_retry_24": f"a7-ipv4-v13-{seed}-relay-retry-24",
                       "ipv4_relay_retry_25": f"a7-ipv4-v13-{seed}-relay-retry-25",
                       "ipv4_relay_retry_26": f"a7-ipv4-v13-{seed}-relay-retry-26",
                       "ipv4_relay_retry_27": f"a7-ipv4-v13-{seed}-relay-retry-27",
                       "ipv4_relay_retry_28": f"a7-ipv4-v13-{seed}-relay-retry-28",
                       "ipv4_relay_retry_29": f"a7-ipv4-v13-{seed}-relay-retry-29",
                       "ipv4_relay_retry_30": f"a7-ipv4-v13-{seed}-relay-retry-30",
                       "ipv4_relay_retry_31": f"a7-ipv4-v13-{seed}-relay-retry-31",
                       "ipv4_relay_retry_32": f"a7-ipv4-v13-{seed}-relay-retry-32",
                       "ipv4_relay_retry_33": f"a7-ipv4-v13-{seed}-relay-retry-33",
                       "ipv4_relay_retry_34": f"a7-ipv4-v13-{seed}-relay-retry-34",
                       "ipv4_relay_retry_35": f"a7-ipv4-v13-{seed}-relay-retry-35",
                       "ipv4_relay_retry_36": f"a7-ipv4-v13-{seed}-relay-retry-36",
                       "ipv4_relay_retry_37": f"a7-ipv4-v13-{seed}-relay-retry-37",
                       "ipv4_cached_r7": "a7-ipv4-v13-887-cached-r7-01",
                       "ipv4_cached_r2_6101": "a7-ipv4-v13-6101-cached-r2-01",
                       "ipv4_cached_r3": "a7-ipv4-v13-157-cached-r3-01"}
    if segment in cohort_services:
        service = cohort_services[segment]
        label = f"gui/{os.getuid()}/com.purellm.{service}"
    else:
        suffix = {"v11": "v11-direct-full", "v6": "v6-full", "v5": "v5-continuation", "direct": ("v9-direct" if seed == 887 else "v8-direct"),
                  "repair": "v10-targeted-repair"}[segment]
        label = f"gui/{os.getuid()}/com.purellm.a7-{suffix}-{seed}"
    return launchd_service_pid(label)


def launchd_service_pid(label):
    lines = command_lines(["/bin/launchctl", "print", label])
    for line in lines or ():
        if match := re.fullmatch(r"\s*pid = ([0-9]+)\s*", line):
            return pid(int(match[1]))
    return None


def mac_report(private, seeds):
    pool = private / "spool" / PILOT / "key_pool/key_pool.sqlite3"
    background = next((row for row in seeds if row.get("segment") == "background"), None)
    if background and isinstance(contract := background.get("contract_sha256"), str) and SHA.fullmatch(contract):
        pool = private / "spool" / contract / "key_pool/key_pool.sqlite3"
    keys, queue, limits = [], [], {}
    try:
        with sqlite3.connect(f"{pool.as_uri()}?mode=ro", uri=True, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            limits = dict(conn.execute("SELECT cap,shared_until,shared_disabled FROM settings").fetchone() or {})
            rows = conn.execute("SELECT seed,round,logical_index,state,key_id,dispatched,response_complete,outcome,usage FROM queue").fetchall()
            for record in conn.execute("SELECT id,present,cooldown_until,disabled_error FROM keys WHERE configured=1 ORDER BY id"):
                alias = record["id"]
                if alias not in KEYS:
                    continue
                state = "available"
                if record["disabled_error"] or not record["present"]:
                    state = "disabled"
                elif any(row["key_id"] == alias and row["state"] == "busy" for row in rows):
                    state = "busy"
                elif record["cooldown_until"] > time.time():
                    state = "cooldown"
                dispatched = [row for row in rows if row["key_id"] == alias and row["dispatched"]]
                used = {name: 0 for name in TOKEN_FIELDS}
                for row in dispatched:
                    for name, count in usage(json.loads(row["usage"] or "{}")).items():
                        used[name] += count
                keys.append({"id": alias, "state": state, "requests": len(dispatched),
                             "complete": sum(row["response_complete"] for row in dispatched),
                             "usage": {field: count for field, count in used.items() if count}})
            queue = [{field: row[field] for field in ("seed", "round", "logical_index", "state", "key_id")}
                     for row in rows if row["state"] in ("queued", "busy") and row["key_id"] in (None, *KEYS)]
    except (OSError, sqlite3.Error, ValueError):
        pass
    for row in seeds:
        contract = row.get("contract_sha256")
        if isinstance(contract, str) and SHA.fullmatch(contract):
            spool = private / "spool" / contract
            if read_json(spool / "run_binding.json").get("contract_sha256") == contract:
                row["provider"] = local_progress(
                    spool, row["seed"], row["api"],
                    track_queue=row["segment"] in ("ipv4", "ipv4_extension", "ipv4_continuation",
                                                   "ipv4_r2_extension", "ipv4_r2_continuation",
                                                   "ipv4_json_extension", "ipv4_ws_extension", "ipv4_bounded_sse",
                                                   *RESILIENT_SEGMENTS),
                    readiness_windows=row.get("readiness_windows", 0),
                )
        if row.get("continuation"):
            row["dispatcher_pid"] = dispatcher_pid(row["seed"], row["segment"])
    return {"keys": keys, "queue": queue, "limits": limits}


def duration_text(value):
    return f"{value:.1f} s" if seconds(value) is not None else "—"


def show(view):
    print(f"A7 status  {view.get('checked_at', '—')}")
    for row in view.get("seeds", ()):
        api = row["api"]
        provider = row.get("provider") or {}
        rounds = row["completed_rounds"] if row["completed_rounds"] is not None else "—"
        print(f"seed {row['seed']}  L{row['labels'] or '—'}/L{row['target_labels']}  "
              f"rounds {rounds}/{row['target_rounds']}  {row['phase']}  worker {row['worker_pid'] or '—'}  "
              f"coordinator {row['coordinator_pid'] or '—'}  Mac {row.get('dispatcher_pid') or '—'}  {row['run'] or '—'}")
        if limits := row.get("normal_api_limits"):
            print(f"  常规选点每轮最多 {limits['logical_per_round']} 个逻辑请求，每个最多 "
                  f"{limits['physical_per_logical']} 次提交，本段总上限 {limits['physical_total']} 次。")
        if limits := row.get("remaining_api_limits"):
            print(f"  R{limits['round']} 恢复每轮最多 {limits['logical']} 个逻辑请求，每个最多 "
                  f"{limits['physical_per_logical']} 次提交，本段总上限 {limits['physical_total']} 次。")
        if repair := row.get("targeted_repair"):
            print(f"  R{repair['round']} 定向修复 {repair['repaired_references']} 条引用，"
                  f"保留 {repair['choice_count']} 个原选点并已训练至 L{repair['fitted_labels']}。")
        if segment_target := row.get("segment_target_labels"):
            print(f"  本段定向纠错及训练目标 L{segment_target}。")
        if prior := row.get("reviewed_previous_api"):
            last = "最后一段正文" if prior.get("event_type") == "response.output_text.delta" else "最后 SSE"
            errors = " ".join(filter(None, (prior["error_type"], prior.get("provider_error_code"))))
            end = (f"{last} {duration_text(prior['last_sse_seconds'])}" if prior.get("last_sse_seconds") is not None
                   else f"事件 {prior.get('event_type', '—')}")
            label = "上个运行段已提交的请求" if prior.get("pre_post_failure") else "上次尝试"
            if prior.get("no_http_headers"):
                upload = {True: "请求体已发出", False: "请求体发送阶段失败"}.get(
                    prior.get("request_upload_confirmed"), "签名回执摘要未列出请求体发送阶段")
                print(f"  {label} {errors}  {upload}  未收到 HTTP 响应头或 SSE")
            else:
                print(f"  {label} HTTP {prior['http_status']}  {errors}  SSE {prior['sse_event_count']}  {end}  未收到完整模型答复")
            if headers_wait := prior.get("receive_headers_seconds"):
                print(f"  上次尝试等待响应头 {duration_text(headers_wait)} 后超时。")
            if prior.get("output_text_done_seconds") is not None:
                print(f"  正文结束事件 {duration_text(prior['output_text_done_seconds'])}，完整响应终态缺失。")
            if prior.get("response_id_matches_transport_alias") is True:
                print("  失败终态的响应 ID 与该 HTTP 请求的传输别名一致。")
            if counts := prior.get("archived_failure_sse_counts"):
                events = "、".join(str(count) for count in counts)
                print(f"  上个运行段两次提交均以 {prior['provider_error_code']} 结束，分别收到 {events} 个 SSE 事件。")
            elif failure_count := prior.get("archived_failure_count"):
                print(f"  上个运行段的 {failure_count} 次提交均收到同类失败事件，原回执未保存具体错误码。")
            if earlier_code := prior.get("earlier_response_error_code"):
                earlier_events = prior.get("earlier_response_sse_count")
                event_count = f"，共收到 {earlier_events} 个 SSE 事件" if earlier_events is not None else ""
                print(f"  上个运行段更早的一次提交收到 {earlier_code}{event_count}，也未收到完整模型答复。")
            if prior.get("pre_post_failure"):
                print("  上个运行段随后在 TLS 握手时超时，未观察到模型 POST、HTTP 响应或 SSE。")
            if earlier := prior.get("earlier_pre_post_failure"):
                print(f"  更早的 W{earlier['wave']} 的另一次尝试在 TLS 握手时超时，未观察到模型 POST。")
            if inactivity := prior.get("sse_inactivity_seconds"):
                print(f"  超时判据为连续 {inactivity} s 无有效 SSE 事件")
            if hard_deadline := prior.get("hard_deadline_seconds"):
                print(f"  请求达到 {hard_deadline} s 硬截止")
        if row.get("archived_correction_provider_post") is False:
            print("  第 8 轮使用原始归档补丁，导入没有提交新的模型请求。")
        if imported := row.get("cached_import"):
            fit_status = (f"已由归档回答训练至 L{imported['fitted_labels']}" if imported.get("fitted_labels")
                          else "本地归档回答的训练凭据待核验")
            print(f"  第 {imported['round']} 轮{fit_status}。"
                  f"保留 {imported['choice_count']} 个选点及顺序，去除 {imported['removed_reference_fields']} 个冗余引用字段，未提交新的模型请求。")
            if future_round := imported.get("future_starts_at_round"):
                print(f"  冗余字段为 {imported['removed_observed_reference_fields']} 个已测引用和 "
                      f"{imported['removed_candidate_reference_fields']} 个候选引用。"
                      f"R{future_round} 起每轮最多 {imported['future_logical_replies']} 个逻辑请求，每个最多 "
                      f"{imported['future_transport_attempts']} 次提交。")
        if response_headers_seconds := row.get("response_headers_seconds"):
            print(f"  请求发出后等待 HTTP 响应头的上限为 {response_headers_seconds} s。")
        if boundary := row.get("boundary_api"):
            print(f"  本轮授权上限 {boundary['maximum_logical_requests']} 个逻辑请求、{boundary['maximum_physical_attempts']} 次提交。")
            if boundary["maximum_physical_attempts"] == 2:
                print(f"  第二次须有首轮失败的封存回执和同请求重试授权，并重新等待 {boundary['replay_delay_seconds']} s。")
            if boundary.get("replay_requires_verified_response_id"):
                print("  本轮重试也须核对服务商响应 ID，确认是同一响应的失败终态。")
            if future_round := boundary.get("future_replay_from_round"):
                print(f"  R{future_round} 起，经回执核验的同请求重试须等待 {boundary['replay_delay_seconds']} s。")
            if "physical_attempts_reserved" in boundary:
                print(f"  已预留 {boundary['physical_attempts_reserved']} 次，其中 {boundary['physical_attempts_posted_confirmed']} 次确认已提交，"
                      f"剩余 {boundary['physical_attempts_remaining']} 次可预留。")
        if api or provider:
            sealed = ("已回传" if provider.get("delivered") else "已封存待回传" if provider.get("sealed")
                      else "待封存" if provider else "未见当前回执")
            if provider.get("phase") == "queued_for_dispatch":
                sealed = "等待派发"
            elif provider.get("phase") == "waiting_for_ipv4_route":
                sealed = "预检中，尚未提交模型请求"
            print(f"  API {api.get('status', '—')}  {api.get('failure') or '—'}  "
                  f"Mac {provider.get('phase', '—')} ({sealed})  HTTP {provider.get('http_status', '—')}  "
                  f"SSE {provider.get('sse_event_count', '—')}  "
                  f"状态更新 {provider.get('progress_age_seconds', '—')} s 前  tokens {provider.get('usage') or '—'}")
            if skipped := api.get("unsubmitted_reservations_skipped"):
                print(f"  已略过 {skipped} 个未提交到 bridge 的本地预留")
            if api.get("cached_import"):
                print("  本轮读取已归档回答，无新 API 提交")
            if (relay_wait := api.get("relay_retry_wait_seconds")) is not None:
                print(f"  已授权的同请求重试最早 {relay_wait} s 后可派发")
            error_details = [f"{label} {provider[field]}" for field, label in (
                ("error_type", "传输异常"), ("provider_error_code", "SSE 错误码"),
                ("provider_error_status", "SSE 错误状态"), ("conflicting_event_type", "ID 变化事件"),
            ) if field in provider]
            if provider.get("response_id_matches_transport_alias") is True:
                error_details.append("变化 ID 与传输别名一致")
            if error_details:
                print("  " + "  ".join(error_details))
            if provider.get("readiness_windows_used"):
                retry = provider.get("readiness_retry_seconds")
                retry_text = f"  下个窗口最早 {retry} s 后" if retry is not None else ""
                sent = {True: "已提交", False: "未提交"}.get(provider.get("provider_dispatched"), "待确认")
                if (sent == "待确认" and not provider.get("sealed") and provider.get("http_status") == 200
                        and type(event_count := provider.get("sse_event_count")) is int and event_count > 0):
                    sent = "已收到 SSE，回执待封存"
                print(f"  IPv4 预检窗口 {provider['readiness_windows_used']}/{provider['readiness_windows']}"
                      f"  模型请求 {sent}{retry_text}")
            if provider and provider.get("phase") != "waiting_for_ipv4_route":
                terminal = ("—" if provider.get("phase") == "queued_for_dispatch"
                            else "已收到" if provider.get("complete_event_seen") else "未收到")
                elapsed = seconds(provider.get("provider_elapsed_seconds"))
                last_event = seconds(provider.get("last_sse_seconds"))
                last_activity = seconds(provider.get("last_sse_activity_seconds"))
                since_event = elapsed - last_event if elapsed is not None and last_event is not None else None
                since_activity = elapsed - last_activity if elapsed is not None and last_activity is not None else None
                inactivity_text = (f"  距上条有效事件 {duration_text(since_activity)}"
                                   if since_activity is not None else "")
                print(f"  派发前等待 {duration_text(provider.get('dispatch_wait_seconds'))}  "
                      f"API 已用时（含预检） {duration_text(elapsed)}  "
                      f"首段正文 {duration_text(provider.get('first_output_seconds'))}  "
                      f"最近两条 SSE 间隔 {duration_text(provider.get('last_sse_interval_seconds'))}  "
                      f"距上条 SSE {duration_text(since_event)}{inactivity_text}  "
                      f"完成终态 {terminal} {duration_text(provider.get('complete_event_seconds'))}")
            if api.get("selection"):
                print(f"  A7 校验 {api['selection']}  {api['selection_errors'] or '—'}")
    for gpu in view.get("gpus") or ():
        print(f"GPU {gpu} (MiB used, MiB total, utilization %)")
    for process in view.get("gpu_processes") or ():
        print(f"  GPU 计算进程 {process}")
    if view.get("gpus") is None:
        print("GPU 查询暂不可用")
    mac = view["mac"]
    print("共享 Key 池", ", ".join(f"{row['id']} {row['state']} ({row['complete']}/{row['requests']})"
                                   for row in mac["keys"]) or "暂不可用")
    limits = mac["limits"]
    cooldown = int(max(0, limits.get("shared_until", 0) - time.time()))
    if cooldown or limits.get("shared_disabled"):
        print(f"  共享冷却 {cooldown} 秒，已停用 {bool(limits.get('shared_disabled'))}")
    for row in mac["queue"]:
        print(f"  seed {row['seed']}  round {row['round']}  turn {row['logical_index']}  {row['state']}  {row['key_id'] or '—'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="repeat the read-only snapshot")
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--relay-pin", action="append", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote:
        pins = {}
        for value in args.relay_pin:
            if parsed := _relay_pin(value):
                seed, wave, contract, approval = parsed
                if (seed, wave) in pins:
                    pins[(seed, wave)] = ("", "")
                else:
                    pins[(seed, wave)] = (contract, approval)
        print(json.dumps(remote_report(relay_pins=pins)))
        return 0
    source = Path(__file__).read_text()
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
               "--", args.ssh_host, f"{ROOT}/environment/venv/bin/python -B - --remote"]
    if args.watch is not None and not 3 <= args.watch <= 300:
        parser.error("watch interval must be between 3 and 300 seconds")
    while True:
        try:
            pins = discover_relay_pins(Path.home() / ".config/purellm", args.ssh_host)
            remote_command = " ".join([command[-1], *(f"--relay-pin {pin}" for pin in pins)])
            result = subprocess.run([*command[:-1], remote_command], input=source, capture_output=True, text=True, timeout=30, check=True)
            view = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            view = {"remote_unavailable": True}
        view["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        view["mac"] = mac_report(Path.home() / ".config/purellm", view.get("seeds", ()))
        if args.json:
            print(json.dumps(view, ensure_ascii=False, indent=None if args.watch else 2), flush=True)
        else:
            show(view)
            print(flush=True)
        if args.watch is None:
            return int(view.get("remote_unavailable", False))
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 130



if __name__ == "__main__":
    raise SystemExit(main())
