# Standard gomoku strategy — the "rule of priorities" (reference)

> **Status: LIVE reference** *(2026-07-04)* — human vocabulary ↔ Allis mapping.

External reference page distilling a clear presentation of **conventional, known**
gomoku strategy. This is *evidence* (someone else's teaching of standard theory),
**not our own synthesis** — it captures how the human gomoku community organizes
move-selection, so a future session can use the standard vocabulary and contrast it
with our own [molecule ⊋ line](idea-pile.md) framing. Where this page editorializes
(the "**Our read**" callouts), it is clearly marked as *our* connective tissue, not
the video's claim.

**Source (provenance).**
- **Video:** *"How to play Gomoku? — The rule of priorities | The base of all Gomoku strategies."*
- **Channel / author:** **GomokuTV** (the host introduces himself by first name —
  rendered "Łukasz" in the auto-captions; surname not reliably recoverable).
- **URL:** <https://youtu.be/1boqoa2rQfU>
- **Jason's framing:** "this is standard gomoku strategy presented quite clearly" —
  treat it as a primary source on conventional tactics / opening theory.
- **Ingest caveat.** Gemini's `analyze_youtube` was **down** (Google "IneligibleTier"
  auth error on the local CLI); this page is built from the **raw caption transcript**
  (`youtube_transcript`, `[MM:SS]` markers below). The video shows on-screen *diagrams*
  of each shape that the captions only point at ("on the screen you can see…"); those
  pictures are **not** captured here — only the spoken theory. Treat shape geometry
  details as cross-checked against [allis-threat-theory.md](allis-threat-theory.md),
  not read off the video's pictures.

---

## 1. The thesis — "the rule of priorities"

The video's entire spine is one idea: **every legal move belongs to a *depth*, and the
farther the depth, the lower the move's priority.** You always answer (or play) the
shallowest-depth thing on the board first. The host states it as the "essence of how
your thinking process in gomoku should look" — *"the rule of priorities is merely an
illustration of the logical loss of gomoku"* `[0:31]` — i.e. it is a teaching scaffold
for the game's forcing logic, not a substitute for style and practice.

The motivating example `[0:00]`: **a five-in-a-row has priority over a four.** Making a
four is pointless if the opponent can complete a five on the reply — the deeper move
(your four) loses to the shallower move (their five). Priority = who is forced first.

> **Our read.** This *depth = priority* ladder is exactly Allis's **strength
> asymmetry** ([allis-threat-theory.md](allis-threat-theory.md) §1): a four has **one**
> forced refutation, a three has **two–three**, a two has many — so "shallower" literally
> means "fewer ways for the opponent to wriggle out." The video teaches the *operational*
> version of the same fact the formal theory proves.

## 2. The five depths (the priority ladder)

The video divides moves into **five depths / priorities**, shallow (highest priority)
to deep (lowest). The table is the takeaway; prose follows.

| Depth | Priority | Contents | One-line meaning |
|---|---|---|---|
| **1** | highest | **overline**, **five-in-a-row** | the win condition (and its 6+ overshoot) |
| **2** | | **four**, **VCF** | a forced win-next-move threat / a forced chain of fours |
| **3** | | **three**, **fukumi**, **VCT** | a must-answer threat / an indirect VCF-threat / a forced chain of threats |
| **4** | | **two**, **VC2 (vc-twos)**, **yobi** | option-building connections that feed VCTs |
| **5** | lowest | **shichō-win (sh-win)**, **positional play**, **"ear-reddening" moves** | theoretical/positional advantage, tempo, open-area shaping |

### Depth 1 — overlines and five `[1:36]–[2:39]`
- **Five-in-a-row** = the win condition; **if you can make exactly five, always do so.**
- **Overline** = **six or more** stones in a row. Because the win is *exactly* five,
  freestyle treats an overline as a non-win; **in principle avoid making overlines for
  yourself**, *but* an overline can be used **to force the opponent** when you are already
  attacking. (The video defers the offensive use of overlines.)

### Depth 2 — fours and VCF `[2:39]–[4:13]`
- **Four** = four stones in a row; **the opponent must block it immediately.** A four
  **open from both sides** (Allis's *straight four*) **wins** — you complete the five next
  move unless the opponent has a higher-priority win.
- **Caveat the video stresses:** play a four **only when it is a *necessary* move** —
  a gratuitous four wastes your own resources and hands the opponent free stones.
- **VCF = Victory by Continuous Fours.** A winning combination of **consecutive fours**:
  every four forces a block, so the opponent must answer each one, and the chain ends in
  a **straight four / a double-four (4×4)** → five. **Same priority as a single four.**
  - **Simplest VCF = a 4×3** (a four-three: one move makes a four *and* a three).
  - **The "cut" `[3:42]`:** the opponent may **make his own four while blocking one of
    yours** — this is called a *cut*, and it can break your chain. So a VCF must be
    **verified to actually win** and **played in the correct order.**

### Depth 3 — threes, fukumi, VCT `[4:13]–[5:46]`
- **Three** = three stones in a row; **the opponent must block it** unless they have a
  higher-priority win. An **unblocked open three extends to a straight four** next move →
  five. Same "only when necessary" caution as fours.
- **Fukumi** = a move that **creates a VCF *threat*** — an **indirect** attack. It does
  not itself force, but it manufactures the *possibility* of winning by VCF on later
  moves. (The standard term for a "VCF is now latent here" preparing move.)
- **VCT = Victory by Continuous Threats.** A winning combination of **consecutive threes
  and/or fukumi** that leads to a straight four, a 4×4, or a VCF → five. Works only if the
  opponent has no higher-priority win.

### Depth 4 — twos, VC2, yobi `[5:46]–[7:21]`
- **Two** = two stones in a row. A single move that **creates several twos at once**
  builds **VCT threats**; the more such threats created at once, the harder the defense.
- **VC2 / "vc-twos"** = a winning combination of **consecutive twos** that leads into a
  VCT. Threes and fours are *inherently* part of VC2 chains; the twos' job is to **extend
  the attack by creating new connections.**
- **Yobi** = a **connecting move for twos**, "usually non-obvious," that gives **future**
  offensive possibilities (not necessarily a direct VCT threat). Play a yobi **only after
  confirming the opponent has no win of the same or higher priority**, and aim it to
  **extend your offense in the most convenient direction.**

### Depth 5 — sh-win, positional play, "ear-reddening" moves `[7:21]–[9:24]`
- **Shichō-win / "sh-win"** = a position where a color has a win of priority **lower than
  VC2**. Examples the video gives: **holding the black stones in standard gomoku is a
  *theoretical* sh-win for black**; or holding the winning color in a "sh-win scheme"
  **within the swap2 rule** (shown as a "white sh-win opening" in the video).
- **Positional playing** = the **first phase of the vast majority of games** (the
  exception being theory-opening games where the opening is dictated and the players fight
  for **tempo**). 
- **Tempo** = momentum / initiative — *"the ability to lead the game and put your opponent
  under pressure and limit his possibilities,"* the ability to control how the game develops.
- **"Ear-reddening" moves** `[8:52]` = advanced **positional moves in an open area with no
  direct connection** (the video calls them by a term borrowed from Go's famous
  *ear-reddening move*). Cited worked example: a **world-championship final** (caption
  garbles the names as "Rudolph Dubsky" vs "…llo"). The host flags these topics as complex
  and **defers them to a separate video** — so this page has the *name* but not the method.

## 3. Glossary (the video's own terms → standard / our terms)

| Video term | Means | Maps to |
|---|---|---|
| **rule of priorities / depth** | move-ordering by how forcing the move is | Allis strength asymmetry ([allis-threat-theory.md](allis-threat-theory.md) §1) |
| **overline** | 6+ in a row (non-win in freestyle) | freestyle overline rule |
| **four** (open four) | 4-in-row, must block; open = won | Allis *four* / *straight four* |
| **three** (open three) | 3-in-row, must block; extends to straight four | Allis *three / broken three* |
| **two** | 2-in-row option-builder | the "2s for options" intuition in [idea-pile.md](idea-pile.md) #10 |
| **VCF** | Victory by Continuous **Fours** (forced four-chain) | Allis winning threat **sequence** of fours (OR-only) — `gomoku/vcf.py:solve_vcf` |
| **VCT** | Victory by Continuous **Threats** (threes+fukumi chain) | Allis winning threat **tree** of threes/fours (AND/OR) — `solve_vct` |
| **fukumi** | a move that *creates a VCF threat* (indirect attack) | a preparing move that makes a VCF latent |
| **yobi** | non-obvious connecting move for twos, future offense | shaping / "2s for options" |
| **VC2 / vc-twos** | winning chain built from consecutive twos | deeper threat-space combination |
| **cut** | opponent makes his own four while blocking yours | Allis *counter-four / conflict* breaking the line |
| **4×3 / 4×4** | four-three / double-four (one move, two shapes) | the double-threat **fork** (Allis §4) |
| **sh-win (shichō-win)** | a win of priority below VC2 (theoretical/positional) | first-player-win theorem ([allis-threat-theory.md](allis-threat-theory.md) §6) |
| **tempo** | initiative / momentum / forcing the opponent | sente |

## 4. How this aligns with — and differs from — our framing

This is the load-bearing part for our project. **The video is a crisp statement of
standard, line-and-threat-centric theory** — which is exactly the *one species* of
structure our [idea-pile.md](idea-pile.md) **#10** thesis says is **not the genus.**

**Where it ALIGNS with us:**
- The **priority ladder = Allis's forced-refutation count.** Depth-2 fours (one block),
  depth-3 threes (two–three blocks), depth-4 twos (many) — the video's operational
  ordering *is* the formal strength asymmetry. Standard theory and Allis agree.
- **VCF / VCT are the video's named win-engines** and are exactly our
  `gomoku/vcf.py` objects ([allis-threat-theory.md](allis-threat-theory.md) §5): VCF =
  OR-only forced-four line; VCT = AND/OR threes+fours tree. Same things, community names.
- **"Twos for options" is a real human primitive.** Depth-4 (twos, yobi, VC2) is the
  video *explicitly* teaching that you build twos because they **create future
  connections / offensive possibilities** — the precise intuition idea #10 quotes
  ("you build 2s because 2s give you options"). The standard theory **has** a shaping
  layer; it just names it within the line vocabulary.

**Where it DIFFERS / what it can't see (our read, not the video's claim):**
- **The whole ladder is line-organized.** Every depth-1–4 object — five, four, three,
  two — is *"N stones in a row."* The framework is **defined on the 4 line directions**,
  so by construction it cannot name a **non-line molecule**. This is the
  [the-claw.md](the-claw.md) blind spot stated from the *human-theory* side, not just the
  Rapfi-engine side: **standard theory is as line-shaped as the engine that learned from
  it.** A density-1/5 knight's-move defensive crystal has **no depth** on this ladder at
  all — it never makes a "row."
- **Depth-5 is where standard theory *gestures at* the non-line world but has no vocabulary
  for it.** "Positional play," "tempo," and especially **"ear-reddening" open-area moves
  "with no direct connection"** are precisely the **shaping / field** offense idea #10
  hunts — moves whose value is relational, not a line. The video **defers these to a future
  video** and treats them as the hardest, least-formalized layer. **That is the tell:** the
  line ladder is fully formalized through depth 4, and the moment value stops being a line,
  the standard theory runs out of names. Our #10 bet is that *that* deferred layer
  (offensive **fields**, catalytic shaping, zero-line-content residuals) is exactly where
  the unmapped strength lives.
- **"Molecule ⊋ line," restated against this source:** the video proves the *subset*
  direction by example — fours/threes/VCF/VCT are real, useful, and *line*. It is silent on
  the *proper* part (⊋): the molecules with no line content. The contrast is the value —
  **standard theory is the complete map of the line species; we are deliberately hunting
  the genus it can't draw** ([idea-pile.md](idea-pile.md) #10, [the-claw.md](the-claw.md)).

## 5. How to use this page
- **As a vocabulary key.** When a human source, a Rapfi log, or a derby idea says
  "fukumi," "yobi," "VC2," "cut," "sh-win," map it here, then to
  [allis-threat-theory.md](allis-threat-theory.md) for the formal object.
- **As the "standard baseline" in the #10 contrast.** When arguing that a representation
  sees *more* than line theory, this is the concrete statement of what "line theory" *is* —
  cite depth 1–4 as the line ladder and depth 5 as the un-formalized field layer.
- **Don't over-trust the geometry.** Shape pictures weren't captured (transcript-only
  ingest); use Allis §1 for exact cell-counts, not this page.

**See also:** [allis-threat-theory.md](allis-threat-theory.md) (formal threat taxonomy +
VCF/VCT), [the-claw.md](the-claw.md) (the canonical non-line molecule the line ladder can't
see), [idea-pile.md](idea-pile.md) #10 (molecule ⊋ line — the framing this page is the
standard-theory foil for), [swap2-opening-protocol.md](swap2-opening-protocol.md) (the
swap2 "sh-win scheme" the video's depth-5 references).
