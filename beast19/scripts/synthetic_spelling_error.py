import sys
import random

def introduce_errors(s,
                     corruption_rate=3e-3,
                     infill_marker="???",
                     max_infill_len=8):
  """
  Artificially add spelling errors and infill markers.
  This function should be applied to the inputs of a correction model.
  The artificial errors are particularly useful to train a network to
  correct spelling when the training data does not contain many
  natural errors.
  Also replaces some substrings with an "infill" marker.  e.g.
  "the fat cat sat on the mat" -> "the fat ca??? the mat"
  This causes the trained model to learn infilling (predicting what text
  to insert at the current cursor position).
  Args:
    s: a string (the uncorrupted text)
    corruption_rate: a floating point value.  Probability of introducing an
      error/infill at each character.
    infill_marker: a string
    max_infill_len: an optional integer - maximum number of characters to remove
      and replace by an infill marker.  None means no infilling.
  Returns:
    a string
  """
  # num_errors = 0
  ret = []
  operations = [
      "delete",  # delete a character
      "insert",  # insert a random character from the input string
      "replace",  # replace a character with a random character from
      #   the input string
      "transpose",  # transpose two adjacent characters
  ]
  if max_infill_len:
    operations.append("infill")
  pos = 0
  while pos < len(s):
    if random.random() >= corruption_rate:
      ret.append(s[pos])
      pos += 1
      continue
    # num_errors += 1
    operation = operations[random.randint(0, len(operations) - 1)]
    if operation == "delete":
      pos += 1
    elif operation == "insert":
      ret.append(s[random.randint(0, len(s) - 1)])
    elif operation == "replace":
      ret.append(s[random.randint(0, len(s) - 1)])
      pos += 1
    elif operation == "transpose":
      ret.append(s[pos + 1] if pos + 1 < len(s) else "")
      ret.append(s[pos])
      pos += 2
    else:
      assert operation == "infill"
      ret.append(infill_marker)
      pos += random.randint(0, max_infill_len)
  return "".join(ret)

for line in sys.stdin:
  print(introduce_errors(line.strip()))
