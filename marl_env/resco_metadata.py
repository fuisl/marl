"""Vendored RESCO signal metadata for supported benchmark maps.

The payload is stored as compressed JSON to keep the source compact while
remaining fully local and deterministic at runtime.
"""

from __future__ import annotations

import base64
import json
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any


_RESCO_SIGNAL_METADATA_B64 = """
eNrtXU2P2zjS/i/21QnIokRKubVlXXXxQYdBYPROerMBku6g04t3gEH++0tSIllFldR2Wz1f28DOxnpIifJT5FMflNy/bz4/fvlU/FZsPvy++f6f2x93p++3Xx5/bD788svm+O642W26d93m484dtf6oH49cm8PckeuT2nr7727Tjm2979mOV2n9VcLR0NP1+Phxt3Ejn54eTre/Pp2+3X7ffLj/79evu82NcDf39fb+7vTj7umHOzja8z78srmRN+Ik7JX87QRADkAbAXBAb4e0wF6EU/rhGh6QA3CMgD+lG67xr4enp4dv8bxuuFBC5YD2FPVXaIdLfr3791M8vx2uGjA5YB3G7Lk/d5tPD/93/+Pp8e72m/vKtoP9No47+2EvHPmBoH74YE/595ff7j6dnr58+3L/2dmw2MEO/fcx9Bgs/fD46e7x9OXTb5sPwp5srz7HM9xIyrMDCM8OIDzLcErgWYZTAs8ynDLyfCPCKSPFHsDseiAnNg6EiI1jIWLH4XhiIRArB2LdpFuPWJglVt0AJdYBhFgHEGIhnBKIhXBKIBbCKV1cBECJleGULi4kmBAbB0LExrEQseNwPLEqEAuBWLkisWqO2KeH7+JGEWpHCJM7QoReFU4L9KpwUqBXhVO6OPcVpRfCKV1cP2pCbxwI0RvHQvSOw3H0Dsvf06sCvbAevftZ4d3LPRVeD2BqPYCJbcSeCq8HMLEemApvHIoIbxyPCG8cdOT5Jg7aBo3ZU8m9GQdlZ+8+6G2T663Th6vpndXbPeyp3nqA0OsAQq/cU731AKFX7qne7sWe6q0HMKkeIHzGUQKfcZTA5zgKz2eQ2SbI7H6UWScLV/M5K7N7tacy6wHCpwMIn7CnMusBwifsqczaKU9l1gOET7mnMnsTRwl8xlECn+MoPJ9BXZugrnsZ+ITr+VxSV7mfqKuDMnV1EGFV7am6eoCwqvZUXe1Mp+rqAcIq7Km63sRRAqtxlMDqOMqyqDZBVPcQWFVXs9rMimojGyqqHsCMegDzeRANFVUPYD49MBVVaDhRhYYTVWioqO7joG0QkoaK6n4clJ20TRDVw0RU99eLajMrqg00VFQ9QOh1AKFXNlRUPUDolQ0V1UY0VFQ9gEn1AOEzjhL4jKMEPsdReD6DqB6CqDZBVPfXi2ozK6qNaqioeoDw6QDCJzRUVD1A+ISGiqqd8lRUPUD4lA0V1X0cJfAZRwl8jqPwfAZRPQRRbYKo7q8X1WZJVKGZiKqDMlF1EGFVNVRUPUBYVQ0V1SaOFFiN43RxtVBR3cdRAqtxlMDqOMqyqB6CqDZBVPfXi+phVlQP8kBF1QOYUQ9gPh+/fP7PkzhQYY0g5jWCU4FVB05g1YETWHWgAtvEwdsgKgcqsM04KDuBD6PADpwTgW2uF9jDrMAe4EAF1gOEagdMqJYHOaVaHuSUanmgYnsQByq2HsAEe4BwG0cL3MaRArfjKDy3kHE7eLF+UImruZ0V24M6ULH1AOHWARNu4QBTbuEAU27hQIXXLgsqvB4g3MoDFd4mjha4jSMFbsdReG7VhNsgvM31wntYEl51mAivgzLhddCE4XgqZjiejBmOp3dxNVAR9gBhGA5UhJs4WmA4jhQYHkdZFGHEcBDh5noR/nZ7f/v57tvd/ZMbz0rbyZUvTt8+Pw6J9mY3pHs+R7nxN+06eW5CLx+6+CXlbd/4Xpb6rJe7a3/rfs7Yex97kQFdu4/cfZpkQ/eP5CZP93f2mv96ePzB3S69s+zq3J2TC+T3zH2J50bIvw+9wI4O+PGn/W63j093j19uv55Vzqdl+bxkH8vytGQ/X6S/f5r1C/dP5ck2kwWWYe0EG5fY/RNgrGewI8YkWl/332XqNy6xDOsn2LjQLKYx2HJgR0A5I2v2a40Bpb3JVyjbu6vOMu9uDTLmCdZOsMi8wljPYEeMUeYh9YvME6yfYIH5J4mxlsE6jM3TriPtKk803Wy9nnc1z7s5OV4o7wRrJ1jkvcBYz2BHjFHeVeoXeSdYP8FavIJUxjvBOozN824i78WUd1iB92Ke9+rkeKG8E6ydYIH37yXGegY7YozyXqR+kXeC9ROsxeunyHgnWIexed6r2bDfTdXraS/naa9PTrwp7QRrJ1ic7hpjPYMdMUZo9xJQUtop1k+wJPAlBlsO7Ag4z3wdZ7wek1snMCsqvJ6nXoqT44ZyT8F2Ckb2DcZ6BjtijLIPqV+HxUFn7COsxe5fZ5OeYB3G5qmXInJvIvchqHVz9nryzQL5bmKZnHwCtlMwkl9hrGewI8Yo+Sr167BCmIx8hLV4KZmMfIJ1GFsgX0byq0i+iuTrFcivFsh386rKySdgOwWj3GuM9Qx2xBglv0j9Ouy2q4x8hLV4JVUZ+QTrMLZA/qQUMTjcflwM13NfL3DvplWdc0/AdgrGie+lqM5mPgWPBKT0l6ljh4WizuhHWJL9AoMtB3YEXLCAitNfijj/yzWFX4oFG7j7kyI3AkVbBk1mkATtWfRIUWoJjbp2WDYS2E/BFscIUmSLgYIdAReMUSRjyGgMHRdEvcKCkAs5rg8SZJ7lZmjLoMkaQNCeRY8UpdYwqGuHdURmGS8BW7LWpMxTL4p2FF0wSEp7JUSDmGgQKdawyELuO+TlefaboS2DRhdhCNhz4JGA1BwV6tlhny+zNJiALV1xk0yYoh1FF8yhp+6iSsZYIx+W8wnx92HCZilxjrYMmpZHQdCeRY8UpfaoUdeOCEuWHhMweQ5F0JZFO4rK57bDRuE6xjxiPddRLFjDT5citwZFWwZN1igJ2rPokaJZ7iZQ347qSpHnbxhtSbwh89Q5QzuKnmOQMiVxIq2QNVJoWS7YxH/HMrcJRVsGTTbRBO1Z9EjRzCYS9e2ouExyaoy2dOWVuU0o2lH0HJugxFommxRr2GQ+uf7udSBLrjOwnYLRf1QE7DnwSMDMGoC6djRMyHNsgrZ05eV5doZ2FJVnbx6Njr0Pq+Z1N5CGksqQ3A+BnYu1ua2YIQcdsqHBybm4g9uDGWfT+EXGaea/C7MBM+jzKAqjbntd+KfvK/368PXh8/2d5DaVZreKnnkPhL4xQt8mmX3vZH776dev//3xZOeSKo2sjP2nLovZaASMlKUEtaXhCILdpJj2aie90mJ/p0CoSpR6SyMUiu/IMd4hnuCjCoAqoAYltzSsJ/AOH+L94xwedQEqWVcVyC2NZRDsv/+kVzfpdaZSPLffBVYdqh3sxn8vUQk3e8eJzM+BpcWZTsbzXP3p70L5tuPSpuvv8dtCIcvKVFoXJyiLotLW3v67a+3/Md692VhQ7DY2+pC7jdh8gN3GrmZlv7PSQlRndIG8i20ET9pF98EtRyVr0LKs1TZztaTBTUmuZ8v0RMvS3pbWoi7LfF3Shh0FyMqcNISlOeJmSwsgFN+RY7I6czwsT0u3rKSotrQyQvEdOcaPgGT4cmg12n/9bek0sTibF6IoFdRbGu5O0JZBp5bNKli0RVLbZnWtvG+y7rtC1lLIqtzSDJ7BexYP1owD0Io7xXfkmFiT4mdYM3/I4JIluorV52oy0dDRttGcvPXMxHpm1np0f+SdVEJqASJfnFxDzzdMDJilmlnDjgK8Dc/IQJn923EtrRtfI88Zp8040BiHZn71jOlznrONF6fjhA7YF1d/MV/s2zrUNozQjkdHf14afTFy/X2jLIN1Bf5Lig8imFzLKMFQgpAGKl83EN79Wicskd+Voq5KU4gS4kctra2MMnUt0HnSu+3Ce/Zy8OwaCtC6LvJOlfPtLgIUEkQ5lPVwez22l9renarqSWwgxw5aSmGg5sIL1zzepGHbnZogfjgv8g6UrrSooqc8Bl2Y4C2LZ+ITlQW7gVrpQuY+gIL9FGzDddogdIMW8OXZZObLAueLlj+ZSzydlTZlKcWEzgnesnhwzZWRukpCGpR9Ah85eCReKiVcqlFmmQ+D9yzexrkAlU0J6iyaYvCOwVlznbv2BqccV9lQNYmTejXDXiAFy2bPtq+4hpZvGC0PUEtT2OSQFiGn8JGDQ1wdLk5rw1O45+BodqjLujDltsjMPsE7Bj+zAobX7UrGxJrMFi9qI6xFBX3wYYLy0dVU4JCoeSaLusgWFsZxmhKWLakuJkJJMy960b3MxT2J3qu1D7uypdA0MJkmVKj75HjMQ6zqV0JUVO1wkJp16PAisk6DupYc7SkqX+Bcghce5Ch57WEOI3lazdGguIBXHBs+GtBkAwrLC2mORNvpUNQS6tyvMA1HviGb85HgKaebaDW61UrhSwP5NA2v5hiFVizFrvJXVhDT0aDpU7xl8ZF1aeN9ULUut/RxHYrvyDGunkzwOPvjespiK6ah5xuC1Kcxsq0O2uJzSa5vx/U92wUgFfMmjkHt1SbG4THrCoL/zOrYE7jl4ImBs53DrGFHAd7G2Ybiuzhmts/LNfR8A2PkatbI1VDKI4icMXL1fDkFKefawfhsLp6sviNSje4FT7mo3jSXRkEmjlHOjRTXy+MtNw9ffzzdfnqSz6fyNF1/pgRON37AfoXKGrY6xe8IQlixFq7kMn6sZjNJKWxHocoivPN3zNEdOgImxtpYlmutSmVsnKQ2Z4Vc+BS/5YKOYTP1Thtpv0QpCxnjABpzpWZYa2tmnUl9qZUu2LRJE8z8jWpFy3s60pRGQqFKkT5CKsIoXKRJS9oVGa1glIlbVZna/5c+6lM+iF348aM5Kas5hQQ9U/J5wXJbuGclbDBVuM06ehWrQjJ8rEDHj6qIH+sqfKzBxI8qoYWJ92HiLdnle5KlMKVlyq3q8FGfoCzdrmZpwk3V/snbaqBA+LtXvnYmxl2xgShklooW5oYO1s0oI41NUUXxDBuRg0o5D2UqOM1fK+9dCe0bK1mX4jnajSysY6n0qbCewwhj6lNdSedpkiXGTm5K+DGLsg6fNLIrpNmBJ+PcBGbDGKt6pbY3vhVEfSm+I8fw0izX3kolrTPUo+TiY1Zy30kbcuraet+0zUZVN+/xTMlo3WX6imr+isrCuuDREiabBhTfkWPIjhWKdyd4CHhFYe9ExNnWxwggwIBj2wQrXKMa5qKLU9/r+HZIxzbtcgimkMIVLabpsoz/UpXOUprlVbz6/HoL3V4Wur2ipqxs4b++n18uCVoRkVrWsBXvK3VSPgUjUIHzbNokp71hc6HPcrMXpBFpLvsjwNsfEVWkT4F3sk1t6TFVmLLpELgpidoV7V7MOLkYE81M1PP1aK2HUFCQxj56BHVdF6rIXA6Bd/gQXl5dtwJU1QWokX0CACmB5A1xS8OGa6o2wRHiY9aAuIN6Ji5ZMficMf5fXARWF711w3nW8+nKCLBLfZv5PoT7CAkdq5dOYL8IpN/rjWtiOMSTF8EobtkoWSpXXFZbGB/kS8fs1M06LE7dFdKZbMoiHVvbE66Sei16KwXWn+sYmKBDwE9MJhgHxwoESCmBBscpHEbt/nmhdIij41Jah1fKcAvoEFCgjGBFexWbhZ2Z5dh3IUxaVeL+qIrZXAz+erHf5c76b6LweYU8rvFXmSArrvjXqMaD/Ec/WfdWSf1frqSiBcytztOLL/xKlVO316tUKW3slB6F1ORRyqqWRlWiSu1KyODlBOpjtFp8myNxpJV7otdZN3y0Nq+NtmkjDC8Pz15kPMP+39z3UUoYKKUsZjrE6a8E2HkjlA0olDKWBKURCWYgASJLpqjtCpt5oDR9t8LGuzZdrU+g7Qy0lxU5sSpeUrvnmv3TcRAfsPUju3Yl7L3ZSGnmdDvFobBLKL9jiBYBA9rMWTXecGFloC4qu+rAqMKAqmdIeCuyvxXZ34rsb0X2tyL7P6HIntz6W6n9rdT+p5XaUfj5VnH/oyruVyQpf+MK+3KMvlo0dVUC+FZrf6u1v9Xar6q1XyVua72Ggmsq/GsooGu7WMutxGsav4iSdUjv/Cj3Po5UW03yG66BX+PLr56Ei9C/7YHxZ//O0t/GKZAXZGIJ61rj43rZXD3EphIAwVCoHpLhLYv38Y2CSqqyrtIU6WfwY44DeVWitCJo8E/4dDN4z+LpPYlK1NrYUOY9mBLnQXNtHdd29vw6P9Bcu3SCy53s8nZlBaNtKkCWN4e3OQ5ktY9pp4phYATke1mUmQSg3sD1BvKKGuqtuN6K/KrFYCTIJ8kU71k8/gCUXZk1lIJW2zi8y3F4Nqx9pq48vloTxXlwHNGY61fWzih0sxGfdUlKCaGz6YNx7OUxTkphxoqfsAFAKGH40pM1L9DSWOzms0pyRE6CHFC4khbPK/JuBa2sgXKvWNosWaafTum4NqBVM9KG404JlXLvKNVbUn0h+FBKTMcKTTKMz2W9aN8hCzzWnz90j2MxDHxnZ5KQlXSvlQ1rHgNUY1CDIm/OF1UNloDw8z95NIg7YMeRLiiIGTEOL4/x8E7ObKz37IpfK6Iju0rLJvE7M0Jq2Ga/XYAbJLWAvbQxei4exx14C8B7acoZKwxtsEK0/cze2WAcslLWriXP7dm9/Def3K8K6aLUW8h+ew010HApLgfiCsmbpKkHEOcXLpk7vwl+hi3SlmT+qneaqyskMmnbk3NV2sq8TedKlbkqBm9zHEc6oRJQZNs2DH7McWydgBvrJgpFFgTf1DNNwNQ0ysxmDN7l+OWvQj4z0dcLTPhd6stWUfDJMs9AGJz+tFGIFeptkWkWxnfkmNTBczzPY62zc7+VUEqVpRoRngsm4677+LN2qUZE11jcvF9hiaXHANh8UUmtTeF+q4akiwn2cdmkVzvpxehfxlOM5RBNZxQQfO7g4r3CTvv3pqK5X9YWMhOm/3wsEJ92oH6Inciv8IuS6FGMC9fIu8JV96s6T9IxvMOHQA/VZnmb0n17PUYOO3JMo+cMnxR+DNgUdMgo7bqzU9L+T+vQucv7XF6VYw3VY9Neb6X0RAy7AWTdrM0gFOQ7QAgnxdMczxdPICT8anM6QqEbRS+oxY2epIAxe8HH50cJSM/yctfwfNAKtZD4oBH/czCFKbVNt6vZWifuQcofEmr3sxByWuIiDTsKwEurn8F3x7o3OuZ8PMzvtWEDROViPT37ANZqjn7m6a4Xh8sjzyBDiQkDtNqIGnAlacS1oVkjcu6oAxeJibDlEY/5yEyMYor70eNhRwK3q6x9Zpwz1l1cFn/Cb128Pbn/xz25f9V205pP6u8mf/ABFzrjpgQqXjN/keGCwgpJ80mp5rkiQUoquT8ikeLh3XPRA/JtSGXRyluUwQvfbWD+AAbzrkP+Ryx27J/hOO/PY1z89zfSTdBL//z58/8Bv10LRw==
""".strip()


_RESERVED_MAP_KEYS = frozenset({"phase_pairs", "pair_to_act_map"})


SUPPORTED_RESCO_MAPS = (
    "grid4x4",
    "arterial4x4",
    "cologne1",
    "cologne3",
    "cologne8",
    "ingolstadt1",
    "ingolstadt7",
    "ingolstadt21",
)


def _normalize_map_name_from_net(net_file: str) -> str:
    net_name = Path(net_file).name
    if net_name.endswith(".net.xml"):
        return net_name[: -len(".net.xml")]
    return Path(net_name).stem


def _iter_signal_ids(map_metadata: dict[str, Any]) -> list[str]:
    signal_ids: list[str] = []
    for signal_id, signal_meta in map_metadata.items():
        if signal_id in _RESERVED_MAP_KEYS:
            continue
        if not isinstance(signal_meta, dict):
            continue
        if "lane_sets" not in signal_meta or "downstream" not in signal_meta:
            continue
        signal_ids.append(str(signal_id))
    return signal_ids


def _normalize_phase_pairs(raw_phase_pairs: Any) -> list[list[str]]:
    if not isinstance(raw_phase_pairs, list):
        raise TypeError("RESCO phase_pairs must be a list.")

    phase_pairs: list[list[str]] = []
    for pair in raw_phase_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError(f"Invalid RESCO phase pair {pair!r}.")
        phase_pairs.append([str(pair[0]), str(pair[1])])
    return phase_pairs


def _normalize_pair_mapping_entry(raw_key: Any, raw_value: Any) -> list[tuple[int, int]]:
    if raw_value is not None:
        return [(int(raw_key), int(raw_value))]

    if isinstance(raw_key, str) and ":" in raw_key:
        left, right = raw_key.split(":", 1)
        return [(int(left), int(right))]

    # Upstream signal.yaml contains a malformed inline mapping on cologne8.
    # YAML 1.1 parses entries such as `1:1` as sexagesimal 61 with a null value.
    key_int = int(raw_key)
    left, right = divmod(key_int, 60)
    return [(left, right)]


def _normalize_pair_to_act_map(
    raw_pair_map: Any,
    *,
    phase_pairs: list[list[str]],
    signal_ids: list[str],
) -> dict[str, dict[int, int]]:
    identity = {
        int(idx): int(idx)
        for idx in range(len(phase_pairs))
    }
    if raw_pair_map is None:
        return {
            signal_id: dict(identity)
            for signal_id in signal_ids
        }
    if not isinstance(raw_pair_map, dict):
        raise TypeError("RESCO pair_to_act_map must be a mapping or null.")

    normalized: dict[str, dict[int, int]] = {}
    for signal_id, raw_signal_map in raw_pair_map.items():
        if not isinstance(raw_signal_map, dict):
            raise TypeError(
                f"RESCO pair_to_act_map[{signal_id!r}] must be a mapping."
            )
        signal_mapping: dict[int, int] = {}
        for raw_key, raw_value in raw_signal_map.items():
            for global_idx, local_idx in _normalize_pair_mapping_entry(raw_key, raw_value):
                signal_mapping[int(global_idx)] = int(local_idx)
        normalized[str(signal_id)] = signal_mapping
    return normalized


def _validate_resco_map_metadata(map_name: str, metadata: dict[str, Any]) -> None:
    phase_pairs = metadata.get("phase_pairs")
    if not isinstance(phase_pairs, list) or not phase_pairs:
        raise ValueError(f"RESCO metadata for {map_name!r} is missing phase_pairs.")

    signal_ids = _iter_signal_ids(metadata)
    if not signal_ids:
        raise ValueError(f"RESCO metadata for {map_name!r} does not define any signals.")

    pair_to_act_map = metadata.get("pair_to_act_map")
    if not isinstance(pair_to_act_map, dict):
        raise ValueError(f"RESCO metadata for {map_name!r} is missing pair_to_act_map.")

    for signal_id in signal_ids:
        signal_meta = metadata[signal_id]
        mapping = pair_to_act_map.get(signal_id)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(
                f"RESCO metadata for {map_name!r} signal {signal_id!r} has no valid action mapping."
            )
        seen_local_actions: set[int] = set()
        for global_idx, local_idx in mapping.items():
            if not isinstance(global_idx, int) or not isinstance(local_idx, int):
                raise TypeError(
                    f"RESCO mapping for {map_name!r} signal {signal_id!r} must use integer indices."
                )
            if global_idx < 0 or global_idx >= len(phase_pairs):
                raise ValueError(
                    f"RESCO mapping for {map_name!r} signal {signal_id!r} references "
                    f"global phase index {global_idx}, but only {len(phase_pairs)} exist."
                )
            if local_idx < 0:
                raise ValueError(
                    f"RESCO mapping for {map_name!r} signal {signal_id!r} references "
                    f"invalid local action index {local_idx}."
                )
            seen_local_actions.add(local_idx)

        expected_local = list(range(len(seen_local_actions)))
        if sorted(seen_local_actions) != expected_local:
            raise ValueError(
                f"RESCO mapping for {map_name!r} signal {signal_id!r} must use contiguous "
                f"local actions starting at 0; got {sorted(seen_local_actions)}."
            )

        if not isinstance(signal_meta.get("lane_sets"), dict):
            raise TypeError(
                f"RESCO metadata for {map_name!r} signal {signal_id!r} is missing lane_sets."
            )
        if not isinstance(signal_meta.get("downstream"), dict):
            raise TypeError(
                f"RESCO metadata for {map_name!r} signal {signal_id!r} is missing downstream."
            )


def _normalize_map_metadata(map_name: str, raw_metadata: Any) -> dict[str, Any]:
    if not isinstance(raw_metadata, dict):
        raise TypeError(f"Vendored metadata for map {map_name!r} is malformed.")

    phase_pairs = _normalize_phase_pairs(raw_metadata.get("phase_pairs"))
    signal_ids = _iter_signal_ids(raw_metadata)
    pair_to_act_map = _normalize_pair_to_act_map(
        raw_metadata.get("pair_to_act_map"),
        phase_pairs=phase_pairs,
        signal_ids=signal_ids,
    )

    normalized: dict[str, Any] = {
        "phase_pairs": phase_pairs,
        "pair_to_act_map": pair_to_act_map,
    }
    for signal_id in signal_ids:
        raw_signal_meta = raw_metadata[signal_id]
        signal_meta = dict(raw_signal_meta)
        signal_meta["lane_sets"] = {
            str(direction): [str(lane_id) for lane_id in lanes]
            for direction, lanes in raw_signal_meta["lane_sets"].items()
        }
        signal_meta["downstream"] = {
            str(direction): (None if downstream is None else str(downstream))
            for direction, downstream in raw_signal_meta["downstream"].items()
        }
        signal_meta["fixed_timings"] = [
            int(value) for value in raw_signal_meta.get("fixed_timings", [])
        ]
        signal_meta["fixed_phase_order_idx"] = int(
            raw_signal_meta.get("fixed_phase_order_idx", 0)
        )
        signal_meta["fixed_offset"] = int(raw_signal_meta.get("fixed_offset", 0))
        signal_meta["pair_to_act_map"] = dict(pair_to_act_map.get(signal_id, {}))
        normalized[signal_id] = signal_meta

    _validate_resco_map_metadata(map_name, normalized)
    return normalized


@lru_cache(maxsize=1)
def load_resco_signal_metadata() -> dict[str, Any]:
    payload = base64.b64decode(_RESCO_SIGNAL_METADATA_B64)
    decoded = zlib.decompress(payload).decode("utf-8")
    data = json.loads(decoded)
    if not isinstance(data, dict):
        raise TypeError("Vendored RESCO signal metadata is malformed.")
    return {
        str(map_name): _normalize_map_metadata(str(map_name), map_metadata)
        for map_name, map_metadata in data.items()
    }


def get_resco_map_metadata(*, map_name: str | None = None, net_file: str | None = None) -> dict[str, Any]:
    data = load_resco_signal_metadata()
    resolved = map_name or (None if net_file is None else _normalize_map_name_from_net(net_file))
    if resolved is None:
        raise ValueError("Pass either map_name or net_file to resolve RESCO metadata.")
    
    # Try vendored metadata first
    if resolved in data:
        metadata = data[resolved]
        if not isinstance(metadata, dict):
            raise TypeError(f"Vendored metadata for map {resolved!r} is malformed.")
        return metadata
    
    # Fallback: try runtime inference for grid5x5
    if resolved == "grid5x5" and net_file is not None:
        try:
            from marl_env.grid_metadata import infer_grid_metadata_from_net
            inferred = infer_grid_metadata_from_net(net_file)
            # Normalize and validate the inferred metadata
            normalized = _normalize_map_metadata(resolved, inferred)
            return normalized
        except Exception as e:
            raise RuntimeError(
                f"Failed to infer metadata for grid5x5 from {net_file}: {e}"
            ) from e
    
    supported = ", ".join(sorted(data))
    raise KeyError(
        f"Unsupported RESCO benchmark map {resolved!r}. Supported vendored metadata: {supported}. "
        f"(grid5x5 requires net_file for runtime inference.)"
    )
