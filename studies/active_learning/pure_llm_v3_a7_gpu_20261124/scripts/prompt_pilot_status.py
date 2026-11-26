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
import sqlite3
import subprocess
import time


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
                      "ipv4_relay_retry_10", "ipv4_cached_r7", "ipv4_cached_r3", "ipv4_cached_r2_6101",
                      "ipv4_r8_targeted_repair", "ipv4_r8_cached_correction", "ipv4_887_r9_continuation")
FORMAL_APPROVAL_FLAGS = ("authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training")
RELAY_RETRY_SECONDS = 180
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
UNSTARTED_ANCHORS = {
    6101: "6959c9417166245403dad173ca82347336de53a37d643591ca449b0cf2a929c4",
    12203: "e056a9924b408e8885e7fa619f0ce63a44b8b3b47a2126d57c1fdc80e3df428d",
}
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
                               previous_suffix="resilient-failure-01"):
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
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != TARGET_LABELS
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
    if wave >= 4 or seed in (6101, 887, 2357):
        prefix = old.get("reviewed_retry_chain")
        first_wave = 1
        if (seed, wave) == (157, 8):
            first_wave = 6
            expected = ((CACHED_R3_157_CONTRACT, CACHED_R3_157_AUTH, 1),)
        elif seed == 12203 and wave in (8, 9, 10):
            first_wave = 6
            expected = ((RELAY_12203_WAVE06_CONTRACT, RELAY_12203_WAVE06_AUTH, 2),
                        (RELAY_12203_WAVE07_CONTRACT, RELAY_12203_WAVE07_AUTH, 2),
                        (RELAY_12203_WAVE08_CONTRACT, RELAY_12203_WAVE08_AUTH, 2))
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
                        (RELAY_2357_WAVE03_CONTRACT, RELAY_2357_WAVE03_AUTH, 2))
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
    elif (seed, wave) == (2357, 3):
        pinned_contract, pinned_auth = RELAY_2357_WAVE02_CONTRACT, RELAY_2357_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_2357_WAVE02_ATTEMPT, RELAY_2357_WAVE03_PROOF
    elif (seed, wave) == (2357, 4):
        pinned_contract, pinned_auth = RELAY_2357_WAVE03_CONTRACT, RELAY_2357_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 4, None, RELAY_2357_WAVE04_PROOF
    elif (seed, wave) == (2357, 5):
        pinned_contract, pinned_auth = RELAY_2357_WAVE04_CONTRACT, RELAY_2357_WAVE04_AUTH
        boundary, prior_attempt, wave_proof = 4, RELAY_2357_WAVE04_ATTEMPT, RELAY_2357_WAVE05_PROOF
    elif (seed, wave) == (887, 3):
        pinned_contract, pinned_auth = RELAY_887_WAVE02_CONTRACT, RELAY_887_WAVE02_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_887_WAVE02_ATTEMPT, RELAY_887_WAVE03_PROOF
    elif (seed, wave) == (887, 4):
        pinned_contract, pinned_auth = RELAY_887_WAVE03_CONTRACT, RELAY_887_WAVE03_AUTH
        boundary, prior_attempt, wave_proof = 8, RELAY_887_WAVE03_ATTEMPT, RELAY_887_WAVE04_PROOF
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
    else:
        return None
    if previous_contract != pinned_contract:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-{wave:02d}"
    physical_attempts = wave_proof["newly_authorized_boundary_physical_attempts"]
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor",
                                           physical_attempts, previous_suffix=f"relay-retry-{wave - 1:02d}")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    registered = {(2357, 5): (RELAY_2357_WAVE05_CONTRACT, RELAY_2357_WAVE05_AUTH),
                  (6101, 6): (RELAY_6101_WAVE06_CONTRACT, RELAY_6101_WAVE06_AUTH),
                  (157, 8): (RELAY_157_WAVE08_CONTRACT, RELAY_157_WAVE08_AUTH),
                  (12203, 9): (RELAY_12203_WAVE09_CONTRACT, RELAY_12203_WAVE09_AUTH),
                  (12203, 10): (RELAY_12203_WAVE10_CONTRACT, RELAY_12203_WAVE10_AUTH)}.get((seed, wave))
    if registered is not None and (contract, record_digest(read_json(run / "authorization.json"))) != registered:
        return None
    signed = read_json(run / "formal_contract.json")
    previous_record = read_json(previous / "formal_contract.json")
    previous_proof = previous_record.get("predecessor") or {}
    previous_profile = previous_record.get("profile", {})
    replay_round = start + 1
    if physical_attempts == 2:
        replay_round = start
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    elif (seed, wave) in ((6101, 5), (2357, 5), (12203, 10)):
        if (previous_profile.get("replay_eligible_from_round") != start
                or (previous / f"seed_{seed}/round_{start:02d}/turn_00/attempt_01.started.json").exists()):
            return None
        if (seed, wave) == (6101, 5) and (contract, record_digest(read_json(run / "authorization.json"))) != (
            RELAY_6101_WAVE05_CONTRACT, RELAY_6101_WAVE05_AUTH,
        ):
            return None
        previous_profile = {**previous_profile, "replay_eligible_from_round": replay_round}
    exact = {**wave_proof, "previous_authorization_sha256": pinned_auth}
    if seed == 887:
        if "cached_import" in proof:
            return None
        exact["reviewed_cached_import"] = RELAY_887_CACHED_REVIEW
    if (seed, wave) == (157, 8):
        if "cached_import" in proof or previous_proof.get("reviewed_cached_import") != RELAY_157_WAVE07_CACHED_REVIEW:
            return None
        exact.update(reviewed_cached_import=RELAY_157_WAVE07_CACHED_REVIEW,
                     reviewed_fitted_prefix_files=CACHED_R3_157_FITTED_FILES,
                     public_resource_receipt_sha256=RELAY_157_WAVE07_RESOURCE_SHA)
    if (start != boundary or profile != previous_profile
            or profile.get("replay_eligible_from_round") != replay_round
            or profile.get("audited_future_transport_replay") is not True
            or type(profile.get("sealed_relay_retry_seconds")) is not int
            or profile["sealed_relay_retry_seconds"] != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    checked_budgets = {
        (2357, 5): (2, RELAY_2357_WAVE04_PROOF["execution_budget"]),
        (6101, 4): (1, RELAY_6101_WAVE03_PROOF["execution_budget"]),
        (6101, 6): (1, RELAY_6101_WAVE05_PROOF["execution_budget"]),
        (157, 8): (1, RELAY_157_WAVE07_PROOF["execution_budget"]),
        (12203, 8): (2, RELAY_12203_WAVE07_PROOF["execution_budget"]),
        (12203, 9): (2, RELAY_12203_WAVE08_PROOF["execution_budget"]),
        (12203, 10): (2, RELAY_12203_WAVE09_PROOF["execution_budget"]),
    }.get((seed, wave))
    if checked_budgets is not None:
        old_physical, old_execution = checked_budgets
        if (previous_proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": old_physical}
                or previous_record.get("budget") != old_execution or signed.get("budget") != wave_proof["execution_budget"]
                or profile.get("sse_inactivity_seconds") != 900):
            return None
    observation = {(6101, 3): RELAY_6101_WAVE03_OBSERVATION, (6101, 4): RELAY_6101_WAVE04_OBSERVATION,
                   (6101, 5): RELAY_6101_WAVE05_OBSERVATION, (6101, 6): RELAY_6101_WAVE06_OBSERVATION,
                   (12203, 4): RELAY_12203_WAVE04_OBSERVATION, (12203, 8): RELAY_12203_WAVE08_OBSERVATION,
                   (12203, 9): RELAY_12203_WAVE09_OBSERVATION, (12203, 10): RELAY_12203_WAVE10_OBSERVATION,
                   (157, 5): RELAY_157_WAVE05_OBSERVATION, (12203, 5): RELAY_12203_WAVE05_OBSERVATION,
                   (157, 6): RELAY_157_WAVE06_OBSERVATION, (12203, 6): RELAY_12203_WAVE06_OBSERVATION,
                   (157, 8): RELAY_157_WAVE08_OBSERVATION,
                   (887, 3): RELAY_887_WAVE03_OBSERVATION, (887, 4): RELAY_887_WAVE04_OBSERVATION,
                   (2357, 3): RELAY_2357_WAVE03_OBSERVATION, (2357, 4): RELAY_2357_WAVE04_OBSERVATION,
                   (2357, 5): RELAY_2357_WAVE05_OBSERVATION}.get((seed, wave))
    if observation is not None:
        sources = signed.get("source_sha256") or {}
        same_response_guard = (seed, wave) in ((2357, 5), (12203, 10), (157, 8))
        required_sources = RELAY_2357_WAVE05_SOURCES if same_response_guard else RELAY_RETRY_QUEUE_SOURCES
        if (proof.get("previous_transport_observation") != observation
                or any(sources.get(name) != sha for name, sha in required_sources.items())
                or same_response_guard and profile.get("request_timeouts", {}).get("total_seconds") != 3600):
            return None
    directory = previous / f"seed_{seed}"
    turn = directory / f"round_{start:02d}/turn_00"
    if (seed, wave) in ((12203, 10), (157, 8)) and {path.name for path in turn.glob("attempt_*.json")} != {
        "attempt_00.started.json", "attempt_00.terminal.json",
    }:
        return None
    multiple_previous = {
        (2357, 4): (RELAY_2357_WAVE03_ATTEMPTS, RELAY_2357_WAVE03_TERMINALS, RELAY_2357_WAVE03_REPLAY_FILES),
        (12203, 8): (RELAY_12203_WAVE07_ATTEMPTS, RELAY_12203_WAVE07_TERMINALS, RELAY_12203_WAVE07_REPLAY_FILES),
        (12203, 9): (RELAY_12203_WAVE08_ATTEMPTS, RELAY_12203_WAVE08_TERMINALS, RELAY_12203_WAVE08_REPLAY_FILES),
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
                "execution_budget": {"first_logical_requests": 27, "maximum_logical_requests": 79, "maximum_physical_attempts": 158}}
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


def remote_seed(runs, seed):
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
                    if successor := reviewed_relay_contract(runs, seed, 8, run, contract):
                        ancestors = (run, *ancestors)
                        run, contract, recovery_round, readiness_windows, relay_boundary_not_before = successor
                        segment = "ipv4_relay_retry_08"
            if segment == "ipv4_relay_retry_06" and (cached := cached_r2_6101_contract(runs, seed, run, contract)):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows, cache_import = cached
                segment = "ipv4_cached_r2_6101"
                relay_retry = RELAY_RETRY_SECONDS
                relay_boundary_not_before = None
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
    api = pending_attempt(run, directory, latest,
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
    if (seed, segment) == (157, "ipv4_relay_retry_07") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_157_WAVE07_OBSERVATION[name] for name in (
            "http_status", "error_type", "sse_event_count", "event_type", "last_sse_seconds",
        )}
        reviewed_previous_api.update(complete_event_seen=False, sse_inactivity_seconds=900)
    if (seed, segment) == (157, "ipv4_relay_retry_08") and latest == recovery_round:
        reviewed_previous_api = {name: RELAY_157_WAVE08_OBSERVATION[name] for name in (
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
    boundary_api = {}
    if seed == 2357 and segment == "ipv4_relay_retry_03" and latest == recovery_round:
        boundary_api = {"maximum_logical_requests": 1,
                        "maximum_physical_attempts": RELAY_2357_WAVE03_PROOF["newly_authorized_boundary_physical_attempts"],
                        "replay_delay_seconds": RELAY_RETRY_SECONDS}
    counted_boundary = {
        (6101, "ipv4_relay_retry_04"): RELAY_6101_WAVE04_PROOF,
        (6101, "ipv4_relay_retry_05"): RELAY_6101_WAVE05_PROOF,
        (6101, "ipv4_relay_retry_06"): RELAY_6101_WAVE06_PROOF,
        (2357, "ipv4_relay_retry_04"): RELAY_2357_WAVE04_PROOF,
        (2357, "ipv4_relay_retry_05"): RELAY_2357_WAVE05_PROOF,
        (157, "ipv4_relay_retry_07"): RELAY_157_WAVE07_PROOF,
        (157, "ipv4_relay_retry_08"): RELAY_157_WAVE08_PROOF,
        (12203, "ipv4_relay_retry_07"): RELAY_12203_WAVE07_PROOF,
        (12203, "ipv4_relay_retry_08"): RELAY_12203_WAVE08_PROOF,
        (12203, "ipv4_relay_retry_09"): RELAY_12203_WAVE09_PROOF,
        (12203, "ipv4_relay_retry_10"): RELAY_12203_WAVE10_PROOF,
    }.get((seed, segment))
    if counted_boundary is not None and latest == recovery_round:
        maximum = counted_boundary["newly_authorized_boundary_physical_attempts"]
        boundary_api = {"maximum_logical_requests": 1, "maximum_physical_attempts": maximum,
                        "replay_delay_seconds": RELAY_RETRY_SECONDS,
                        **boundary_attempt_counts(run, directory, latest, maximum)}
        if (seed, segment) == (2357, "ipv4_relay_retry_05"):
            boundary_api["future_replay_from_round"] = 5
        if (seed, segment) == (12203, "ipv4_relay_retry_10"):
            boundary_api.update(future_replay_from_round=6, future_replay_requires_verified_response_id=True)
        if (seed, segment) == (157, "ipv4_relay_retry_08"):
            boundary_api.update(future_replay_from_round=5, future_replay_requires_verified_response_id=True,
                                replay_requires_verified_response_id=True)
    if segment == "ipv4_887_r9_continuation" and latest == recovery_round:
        boundary_api = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2,
                        "replay_delay_seconds": RELAY_RETRY_SECONDS,
                        **boundary_attempt_counts(run, directory, latest, 2)}
    if seed == 157 and segment in ("ipv4_cached_r3", "ipv4_relay_retry_07", "ipv4_relay_retry_08"):
        cache_import = {"round": 3, "fitted_labels": 461, "choice_count": 32, "removed_reference_fields": 79,
                        "provider_dispatched": False}
    if latest == TARGET_ROUNDS:
        phase = "complete"
    elif segment == "ipv4_r8_cached_correction" and latest == 9:
        phase = "cached_correction_round_complete"
    elif segment == "ipv4_r8_targeted_repair" and latest == recovery_round + 1:
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
    elif cache_import and latest == cache_import["round"] + 1 and not worker_live and not api and not paused:
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
            **({"boundary_api": boundary_api} if boundary_api else {}),
            **({"cached_import": cache_import} if cache_import else {}),
            **({"archived_correction_provider_post": False} if segment in ("ipv4_r8_cached_correction", "ipv4_887_r9_continuation") else {}),
            "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
            "completed_rounds": latest, "labels": 333 + latest * 32 if latest is not None else None,
            "phase": phase, "worker_pid": worker_pid, "worker_active": worker_live,
            "worker_error": safe_code(worker.get("error_type")), "coordinator_pid": coordinator,
            "coordinator_active": coordinator_live, "gpu": gpu.get("gpu"), "api": api, **metrics}


def command_lines(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=8).splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def remote_report(runs=RUNS):
    return {"seeds": [remote_seed(runs, seed) for seed in SEEDS], "gpus": command_lines([
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
    if segment == "ipv4_r8_cached_correction":
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
                       "ipv4_887_r9_continuation": "a7-ipv4-v13-887-r9-continuation-01",
                       "ipv4_relay_retry_07": f"a7-ipv4-v13-{seed}-relay-retry-07",
                       "ipv4_relay_retry_08": f"a7-ipv4-v13-{seed}-relay-retry-08",
                       "ipv4_relay_retry_09": f"a7-ipv4-v13-{seed}-relay-retry-09",
                       "ipv4_relay_retry_10": f"a7-ipv4-v13-{seed}-relay-retry-10",
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
        if prior := row.get("reviewed_previous_api"):
            last = "最后一段正文" if prior["event_type"] == "response.output_text.delta" else "最后 SSE"
            errors = " ".join(filter(None, (prior["error_type"], prior.get("provider_error_code"))))
            print(f"  上次尝试 HTTP {prior['http_status']}  {errors}  SSE {prior['sse_event_count']}  "
                  f"{last} {duration_text(prior['last_sse_seconds'])}  未收到完整终态")
            if inactivity := prior.get("sse_inactivity_seconds"):
                print(f"  超时判据为连续 {inactivity} s 无有效 SSE 事件")
            if hard_deadline := prior.get("hard_deadline_seconds"):
                print(f"  请求达到 {hard_deadline} s 硬截止")
        if row.get("archived_correction_provider_post") is False:
            print("  第 8 轮使用原始归档补丁，导入没有提交新的模型请求。")
        if imported := row.get("cached_import"):
            print(f"  第 {imported['round']} 轮已由归档回答训练至 L{imported['fitted_labels']}。"
                  f"保留 {imported['choice_count']} 个选点及顺序，去除 {imported['removed_reference_fields']} 个冗余引用字段，未提交新的模型请求。")
            if future_round := imported.get("future_starts_at_round"):
                print(f"  冗余字段为 {imported['removed_observed_reference_fields']} 个已测引用和 "
                      f"{imported['removed_candidate_reference_fields']} 个候选引用。"
                      f"R{future_round} 起每轮最多 {imported['future_logical_replies']} 个逻辑请求，每个最多 "
                      f"{imported['future_transport_attempts']} 次提交。")
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
    args = parser.parse_args()
    if args.remote:
        print(json.dumps(remote_report()))
        return 0
    source = Path(__file__).read_text()
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
               "--", args.ssh_host, f"{ROOT}/environment/venv/bin/python -B - --remote"]
    if args.watch is not None and not 3 <= args.watch <= 300:
        parser.error("watch interval must be between 3 and 300 seconds")
    while True:
        try:
            result = subprocess.run(command, input=source, capture_output=True, text=True, timeout=30, check=True)
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
