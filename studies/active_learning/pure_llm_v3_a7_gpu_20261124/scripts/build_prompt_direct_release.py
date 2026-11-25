"""Reproduce the single networking change for the isolated one-request direct release."""
import argparse
import hashlib
from pathlib import Path

PROVIDER = Path("studies/active_learning/pure_llm_v3_a7_gpu_20261124/a7_gpu/provider.py")
BASE = "74b66b86a0dc053c11247cbd0a2df9a48124c823355fd3e09c5090700de1d16a"
DIRECT = "a0b4d47dc973aa5368acead4dc0157ac230f91cc39bf78b8794f468bf1623c11"


def provider_source(data):
    if hashlib.sha256(data).hexdigest() != BASE:
        raise ValueError("the direct release must use the exact verified V5 provider source")
    old = b"async with httpx.AsyncClient(timeout=sdk_timeout, follow_redirects=False, event_hooks=event_hooks) as http:"
    new = (b"async with httpx.AsyncClient(timeout=sdk_timeout, follow_redirects=False, event_hooks=event_hooks,\n"
           b"                                         trust_env=False) as http:")
    if data.count(old) != 1:
        raise ValueError("expected exactly one V5 Responses HTTP client")
    result = data.replace(old, new)
    if hashlib.sha256(result).hexdigest() != DIRECT:
        raise ValueError("direct network provider source does not match the approved patch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    target = args.staging / PROVIDER
    if args.staging.resolve() == args.parent_release.resolve():
        raise ValueError("the V5 release remains immutable")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = provider_source((args.parent_release / PROVIDER).read_bytes())
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("existing staging provider differs")
    else:
        with target.open("xb") as output:
            output.write(data)
    print(DIRECT)


if __name__ == "__main__":
    main()
