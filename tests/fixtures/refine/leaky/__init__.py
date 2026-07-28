"""Twenty-seven strategies, each carrying exactly one deliberate defect.

Nine defect categories, three variants apiece:

* ``_obvious`` — the cheat written the most direct way an author would reach for.
* ``_reworded`` — the same cheat with every identifier renamed to something bland. A detector that
  catches the obvious variant but not this one is matching vocabulary rather than structure, which
  is a detector that any author will defeat by accident within a week.
* ``_buried`` — the cheat set inside thirty or more lines of ordinary strategy code, to measure
  whether detection degrades when the defect is not the only interesting thing in the file.

None of these strategies is meant to be sensible or profitable. They are meant to be realistic
enough that the defect is the only thing wrong with them.
"""
