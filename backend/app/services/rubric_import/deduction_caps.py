"""Separate AI mutually exclusive group limits from per-rule repeat limits."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping
from hashlib import sha256


# atomic_rules.mutex_group 的列宽（String(100)）。SQLite 不检查长度，Postgres 超长直接拒绝写入。
MUTEX_GROUP_MAX_CHARS = 100


def suffixed_mutex_groups(mutex, fingerprint):
    """撞名时可用的互斥组名，按优先顺序，每个都不超过列宽。

    - ``原名-指纹前 8 位``：放得下时首选，页面上一眼看得出来自哪一组。
    - ``原名-原名哈希前 8 位-指纹前 8 位``：原名太长要截短时只用它，首选已被占用时也
      退到它。截短会让前缀相同的两个原名变成同一个名字，原名哈希把它们分开。

    起草生成的原名是 ``{评分项}-B{批}-G{组}-SEVERITY``，远短于列宽；接口直接写进来的
    AI 行可以更长。候选名只由原名和指纹决定、不看占用情况，所以追加同步不用重建整份
    投影，就能认出编译时写进细则的是哪一个。
    """
    fingerprint_tail = f"-{fingerprint[:8]}"
    digest_tail = f"-{sha256(mutex.encode('utf-8')).hexdigest()[:8]}{fingerprint_tail}"
    names = []
    if len(mutex) + len(fingerprint_tail) <= MUTEX_GROUP_MAX_CHARS:
        names.append(f"{mutex}{fingerprint_tail}")
    names.append(f"{mutex[:MUTEX_GROUP_MAX_CHARS - len(digest_tail)]}{digest_tail}")
    return tuple(names)


def ai_mutex_identity(row):
    """(互斥组, 生成指纹)；不是可归属的 AI 起草行时返回 None。

    只认归属：来源以 ``ai_`` 开头，并带互斥组、生成指纹和草稿行键。这是
    ``normalize_ai_group_caps`` 识别口径里的归属部分；它另外要求的单次命中、无分档、
    指纹与 ``generation_metadata`` 一致等条件是为了核验组上限，规整互斥组用不上。
    人工、Excel 行和缺少归属信息的旧行都不规整。编译与追加路径的同步共用这一个判断。
    """
    if not isinstance(row, Mapping):
        return None
    mutex = row.get("mutex_group")
    fingerprint = row.get("generation_fingerprint")
    if not (str(row.get("source") or "").startswith("ai_")
            and isinstance(mutex, str) and mutex
            and isinstance(fingerprint, str) and fingerprint
            and row.get("draft_row_key")):
        return None
    return mutex, fingerprint


def effective_mutex_groups(rows, *, compiled):
    """每行结构化扣分真正写进 AtomicRule 的互斥组，与 ``rows`` 等长。

    ``compiled`` 是会编译成细则的行下标（从 0 起）。评分项投影保留起草时的原名，
    只有写进细则的值在这里规整（AI起草互斥组规整与区间拆档方案 §4.1）：

    - 同一次生成的同一组只剩 1 行会编译：不写互斥组。执行器本来就跳过单条互斥组，
      发布校验却会把它当成定义写错。
    - 2 行以上、同名已被投影里更早的组或人工行占用：改用 ``suffixed_mutex_groups`` 里
      第一个没被占用的名字（通常是 ``原名-指纹前 8 位``），不让两次生成的不同问题并进
      同一个互斥组。先应用的那一代保持原名，已确认的细则不受影响。
    - 人工和 Excel 行原样返回，写错了照旧由发布校验挡住。
    """
    result = [row.get("mutex_group") if isinstance(row, Mapping) else None for row in rows]
    groups = {}
    for index, row in enumerate(rows):
        identity = ai_mutex_identity(row)
        if identity is not None and index in compiled:
            groups.setdefault(identity, []).append(index)
    # 人工行的名字不管出现在前还是在后都先占住：它们不规整，AI 组只能让开。
    taken = {
        result[index] for index, row in enumerate(rows)
        if index in compiled and result[index] and ai_mutex_identity(row) is None
    }
    for (mutex, fingerprint), members in groups.items():  # 按首次出现的顺序
        if len(members) < 2:
            name = None
        else:
            name = mutex
            if name in taken:
                candidates = suffixed_mutex_groups(mutex, fingerprint)
                name = next((item for item in candidates if item not in taken), candidates[-1])
            taken.add(name)
        for index in members:
            result[index] = name
    return result


def normalize_ai_group_caps(rows, *, criterion_code, maximum):
    """Normalize only attributable single-hit AI groups, including old clients.

    A group cap is redundant at execution only when every member is single-hit,
    shares the mutex, and each deduction fits the group limit. Keep that limit
    in the criterion projection and original source, never in AtomicRule.cap.
    Unknown/manual repeat caps must still reach publication validation intact.
    """
    result = deepcopy(rows)
    groups = {}
    for row in result:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("draft_row_key") or "").split("::")
        fingerprint = row.get("generation_fingerprint")
        if len(key) == 3 and isinstance(fingerprint, str):
            groups.setdefault((fingerprint, key[0], key[1]), []).append(row)
    for (fingerprint, code, group), members in groups.items():
        valid = bool(fingerprint and code == criterion_code and group)
        mutexes, caps = set(), set()
        for row in members:
            metadata = row.get("generation_metadata")
            valid &= (str(row.get("source") or "").startswith("ai_")
                      and isinstance(metadata, Mapping)
                      and metadata.get("fingerprint") == fingerprint
                      and row.get("repeat_policy") == "once"
                      and bool(row.get("mutex_group"))
                      and not row.get("levels"))
            try:
                cap = Decimal(str(row.get("group_cap_points", row.get("cap_points"))))
                points = Decimal(str(row.get("points")))
                limit = Decimal(str(maximum))
                valid &= all(n.is_finite() for n in (cap, points, limit)) and 0 < points <= cap <= limit
                # Explicit group and individual limits may not disagree.
                if "group_cap_points" in row and row.get("cap_points") is not None:
                    valid &= Decimal(str(row["cap_points"])) == cap
                caps.add(cap)
                mutexes.add(row.get("mutex_group"))
            except (InvalidOperation, ValueError, TypeError):
                valid = False
        valid &= len(caps) == 1 and len(mutexes) == 1
        if valid:
            for row in members:
                row["group_cap_points"] = row.get("group_cap_points", row.get("cap_points"))
                row["cap_points"] = None
        elif any("group_cap_points" in row for row in members):
            raise ValueError(f"{criterion_code} AI 规则组上限无法核验，请检查单次命中、互斥组、来源和分值范围")
    # Explicit new-format group caps without attributable group identity fail closed.
    if any(isinstance(row, Mapping) and "group_cap_points" in row and row.get("cap_points") is not None for row in result):
        raise ValueError(f"{criterion_code} AI 规则组上限与单条累计上限冲突")
    for row in result:
        if isinstance(row, Mapping) and "group_cap_points" in row:
            key = str(row.get("draft_row_key") or "").split("::")
            if len(key) != 3 or not isinstance(row.get("generation_fingerprint"), str):
                raise ValueError(f"{criterion_code} AI 规则组缺少可追溯编号")
    return result
