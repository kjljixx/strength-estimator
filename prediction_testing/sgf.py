from __future__ import annotations


def keep_last_moves(sgf: str, move_count: int) -> str:
  """Return the main line with its root properties and final move_count moves."""
  if move_count <= 0:
    raise ValueError("move_count must be positive")

  nodes = _main_line_nodes(sgf)
  if not nodes:
    raise ValueError("SGF has no main-line nodes")

  root = nodes[0]
  moves = [node for node in nodes[1:] if _is_move_node(node)]
  return "(" + root + "".join(moves[-move_count:]) + ")"


def _main_line_nodes(sgf: str) -> list[str]:
  nodes: list[str] = []
  node_start: int | None = None
  depth = 0
  in_value = False
  escaped = False

  for index, char in enumerate(sgf):
    if in_value:
      if escaped:
        escaped = False
      elif char == "\\":
        escaped = True
      elif char == "]":
        in_value = False
      continue
    if char == "[":
      in_value = True
    elif char == "(":
      depth += 1
    elif char == ")":
      if depth == 1 and node_start is not None:
        nodes.append(sgf[node_start:index])
        node_start = None
      depth -= 1
    elif char == ";" and depth == 1:
      if node_start is not None:
        nodes.append(sgf[node_start:index])
      node_start = index

  return nodes


def _is_move_node(node: str) -> bool:
  return node.startswith(";B[") or node.startswith(";W[")
