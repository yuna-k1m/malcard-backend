from __future__ import annotations


def segments_to_syllable_boundaries(
    segments: list[dict],
    syllable_positions: list[str],
) -> list[tuple[float, float]]:
    """음소 segments → 음절 단위 시간 경계 변환.

    각 segment의 syllable_position("onset" / "nucleus" / "coda")에 따라
    동일 음절에 속하는 segments를 그룹핑한다. 음절 간 묵음(pause)은
    어느 boundary에도 포함되지 않으므로, 발화 사이 긴 간격이 들어가도
    인접 음절 경계가 왜곡되지 않는다.

    Args:
        segments: 각 dict에 'start_time', 'end_time' 필요.
        syllable_positions: segments와 1:1 대응. 각 값은 "onset" / "nucleus" / "coda".
                            보통 src.korean_ipa.pronunciation_to_ipa(text).tokens
                            의 syllable_position 속성에서 추출한다.

    Returns:
        list of (t_start, t_end) per syllable.
    """
    if len(segments) != len(syllable_positions):
        raise ValueError(
            f"segments({len(segments)})와 syllable_positions({len(syllable_positions)}) 길이 불일치"
        )

    # ── 각 segment에 음절 인덱스 부여 ──────────────────────────────────────
    # 음절 시작 조건:
    #   1) 첫 segment
    #   2) coda    → onset/nucleus  (받침으로 끝난 음절 다음)
    #   3) nucleus → onset          (받침 없는 음절 다음, 자음 onset으로 시작)
    #   4) nucleus → nucleus        (받침 없는 음절 다음, ㅇ초성 음절: 예 "아이")
    syl_idx = -1
    last_pos: str | None = None
    syl_of_seg: list[int] = []
    for pos in syllable_positions:
        new_syllable = (
            last_pos is None
            or (last_pos == "coda" and pos in ("onset", "nucleus"))
            or (last_pos == "nucleus" and pos in ("onset", "nucleus"))
        )
        if new_syllable:
            syl_idx += 1
        syl_of_seg.append(syl_idx)
        last_pos = pos

    # ── 음절별 (start, end) 계산 ─────────────────────────────────────────
    # nucleus 없는 음절(자음만 있는 비표준 케이스)은 boundary 생성 안 함
    boundaries: list[tuple[float, float]] = []
    for s in range(syl_idx + 1):
        seg_idxs = [i for i, x in enumerate(syl_of_seg) if x == s]
        if not any(syllable_positions[i] == "nucleus" for i in seg_idxs):
            continue
        t_start = segments[seg_idxs[0]]["start_time"]
        t_end = segments[seg_idxs[-1]]["end_time"]
        boundaries.append((t_start, t_end))
    return boundaries