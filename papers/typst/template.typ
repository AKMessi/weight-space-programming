// Publication-grade academic preprint template (NeurIPS / arXiv aesthetic)
// Styled for New Computer Modern + New Computer Modern Math in Typst 0.15+

#let navy = rgb("#172554")
#let accent-blue = rgb("#1D4ED8")
#let accent-green = rgb("#047857")
#let slate-dark = rgb("#1E293B")
#let slate-muted = rgb("#475569")
#let rule-gray = rgb("#CBD5E1")
#let soft-bg = rgb("#F8FAFC")
#let head-bg = rgb("#F1F5F9")
#let win-bg = rgb("#ECFDF5")
#let warn-bg = rgb("#FEF2F2")
#let callout-blue-bg = rgb("#EFF6FF")
#let callout-green-bg = rgb("#ECFDF5")
#let callout-slate-bg = rgb("#F8FAFC")

#let paper(
  title: "",
  short-title: "",
  author: "Aaryan Kakad",
  affiliation: "Independent Machine Learning Researcher",
  repo-url: "https://github.com/AKMessi/weight-space-programming",
  repo-label: "github.com/AKMessi/weight-space-programming",
  date: "November 2026",
  preprint-tag: "PREPRINT · INDEPENDENT MACHINE LEARNING RESEARCH",
  companion-note: none,
  abstract: [],
  keywords: (),
  body,
) = {
  set document(
    title: title,
    author: author,
    keywords: keywords,
  )

  set page(
    paper: "us-letter",
    margin: (top: 0.86in, bottom: 0.82in, left: 0.78in, right: 0.78in),
    header-ascent: 24%,
    footer-descent: 28%,
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 8.2pt, fill: slate-muted)
        grid(
          columns: (1fr, auto),
          align: (left, right),
          emph(short-title),
          [#author],
        )
        v(-0.62em)
        line(length: 100%, stroke: 0.45pt + rule-gray)
      }
    },
    footer: context {
      set text(size: 8.2pt, fill: slate-muted)
      let p = counter(page).get().first()
      let total = counter(page).final().first()
      line(length: 100%, stroke: 0.35pt + rule-gray)
      v(-0.55em)
      grid(
        columns: (1fr, auto),
        align: (left, right),
        [#preprint-tag #h(0.4em) · #h(0.4em) #link(repo-url)[#text(fill: slate-muted)[#repo-label]]],
        strong[#p / #total],
      )
    },
  )

  set text(
    font: "New Computer Modern",
    size: 9.8pt,
    fill: rgb("#0F172A"),
    lang: "en",
    hyphenate: true,
  )

  set par(
    justify: true,
    leading: 0.62em,
    spacing: 0.78em,
  )

  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    v(1.05em, weak: true)
    block(breakable: false, spacing: 0.55em)[
      #set text(size: 11.8pt, weight: "bold", fill: navy)
      #if it.numbering != none {
        counter(heading).display()
        h(0.55em)
      }
      #it.body
    ]
  }
  show heading.where(level: 2): it => {
    v(0.82em, weak: true)
    block(breakable: false, spacing: 0.45em)[
      #set text(size: 10.4pt, weight: "bold", fill: slate-dark)
      #if it.numbering != none {
        counter(heading).display()
        h(0.45em)
      }
      #it.body
    ]
  }
  show heading.where(level: 3): it => {
    v(0.55em, weak: true)
    block(breakable: false, spacing: 0.35em)[
      #set text(size: 9.8pt, weight: "bold", fill: slate-dark)
      #it.body
    ]
  }

  // Inline and block monospace styling (clean LaTeX \texttt feel)
  show raw.where(block: false): it => {
    text(font: "New Computer Modern Mono", size: 8.5pt, fill: rgb("#1E293B"), it)
  }
  show raw.where(block: true): it => {
    block(
      width: 100%,
      fill: soft-bg,
      stroke: 0.5pt + rule-gray,
      radius: 2.5pt,
      inset: (x: 8pt, y: 6pt),
      breakable: false,
    )[
      #set text(font: "New Computer Modern Mono", size: 7.8pt, fill: rgb("#0F172A"))
      #set par(leading: 0.52em, justify: false)
      #it
    ]
  }

  // Subtle academic link styling
  show link: set text(fill: accent-blue)
  show ref: set text(fill: navy)

  // Figure & caption styling
  set figure(gap: 0.55em)
  show figure: it => {
    v(0.55em, weak: true)
    block(breakable: false, width: 100%)[
      #align(center)[#it.body]
      #v(0.32em)
      #align(left)[
        #set text(size: 8.6pt, fill: slate-dark)
        #set par(justify: true, leading: 0.52em)
        #it.caption
      ]
    ]
    v(0.55em, weak: true)
  }
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): it => {
    v(0.55em, weak: true)
    block(breakable: false, width: 100%)[
      #align(left)[
        #set text(size: 8.7pt, fill: slate-dark)
        #set par(justify: true, leading: 0.50em)
        #it.caption
      ]
      #v(0.28em)
      #align(center)[#it.body]
    ]
    v(0.55em, weak: true)
  }

  // Equation spacing
  show math.equation.where(block: true): it => {
    v(0.35em, weak: true)
    it
    v(0.35em, weak: true)
  }

  // Compact publication bibliography styling
  show bibliography: it => {
    set text(size: 8.35pt)
    set par(leading: 0.48em, spacing: 0.35em, justify: true)
    it
  }

  // ==================== NEURIPS / ARXIV TITLE BLOCK ====================
  align(center)[
    #set par(justify: false, leading: 0.55em)
    #text(size: 7.8pt, weight: "bold", tracking: 0.08em, fill: slate-muted)[#preprint-tag #h(0.6em) · #h(0.6em) #upper(date)]
    #v(0.32em)
    #line(length: 100%, stroke: 1.8pt + navy)
    #v(0.55em)
    #text(size: 15.6pt, weight: "bold", fill: navy, hyphenate: false)[#title]
    #v(0.55em)
    #line(length: 100%, stroke: 0.7pt + navy)
    #v(0.65em)
    #text(size: 11.4pt, weight: "bold", fill: rgb("#0F172A"))[#author]
    #v(0.18em)
    #text(size: 9.4pt, style: "italic", fill: slate-muted)[#affiliation]
    #v(0.18em)
    #text(size: 8.6pt, fill: slate-dark)[
      Code, Raw Evidence & Reproducibility Manifests: #link(repo-url)[#raw(repo-label)]
    ]
    #if companion-note != none {
      v(0.16em)
      text(size: 8.2pt, style: "italic", fill: slate-muted)[#companion-note]
    }
  ]

  v(0.75em)

  // Abstract box
  block(
    width: 100%,
    fill: soft-bg,
    stroke: (left: 2.5pt + navy, rest: 0.5pt + rule-gray),
    radius: (right: 3pt),
    inset: (x: 12pt, y: 9pt),
    breakable: false,
  )[
    #align(center)[#text(size: 9.6pt, weight: "bold", tracking: 0.06em, fill: navy)[ABSTRACT]]
    #v(0.28em)
    #set text(size: 9.05pt)
    #set par(justify: true, leading: 0.56em)
    #abstract
    #if keywords.len() > 0 {
      v(0.45em)
      text(size: 8.4pt, fill: slate-muted)[
        *Keywords:* #keywords.join(" · ")
      ]
    }
  ]

  v(0.55em)

  body
}

// ==================== CALLOUT / THEOREM ENVIRONMENTS ====================
#let env-box(kind-label, title: none, accent: navy, bg: callout-blue-bg, body) = {
  v(0.45em, weak: true)
  block(
    width: 100%,
    fill: bg,
    stroke: (left: 2.5pt + accent, rest: 0.45pt + rule-gray),
    radius: (right: 2.5pt),
    inset: (x: 10pt, y: 7.5pt),
    breakable: false,
  )[
    #set text(size: 9.35pt)
    #set par(justify: true, leading: 0.58em)
    #text(weight: "bold", fill: accent)[#kind-label#if title != none [ (#title)].] #h(0.25em)
    #body
  ]
  v(0.45em, weak: true)
}

#let proposition(num, title: none, body) = env-box(
  "Proposition " + str(num), title: title, accent: navy, bg: callout-blue-bg, emph(body)
)
#let theorem(num, title: none, body) = env-box(
  "Theorem " + str(num), title: title, accent: navy, bg: callout-blue-bg, emph(body)
)
#let corollary(num, title: none, body) = env-box(
  "Corollary " + str(num), title: title, accent: accent-green, bg: callout-green-bg, emph(body)
)
#let definition(num, title: none, body) = env-box(
  "Definition " + str(num), title: title, accent: slate-dark, bg: callout-slate-bg, body
)
#let proof-box(body) = {
  v(0.2em, weak: true)
  block(width: 100%, inset: (left: 4pt, right: 2pt, y: 2pt))[
    #set text(size: 9.35pt)
    #set par(justify: true, leading: 0.58em)
    #emph[Proof.] #h(0.25em) #body #h(1fr) $qed$
  ]
  v(0.4em, weak: true)
}
#let algorithm-box(num, title, body) = {
  v(0.45em, weak: true)
  block(
    width: 100%,
    fill: soft-bg,
    stroke: (top: 1.3pt + navy, bottom: 1.1pt + navy, left: 0pt, right: 0pt),
    inset: (x: 8pt, y: 6.5pt),
    breakable: false,
  )[
    #text(size: 9.2pt, weight: "bold", fill: navy)[Algorithm #num: #title]
    #v(0.22em)
    #line(length: 100%, stroke: 0.5pt + rule-gray)
    #v(0.25em)
    #set text(size: 8.85pt)
    #set par(justify: false, leading: 0.56em, spacing: 0.35em)
    #body
  ]
  v(0.45em, weak: true)
}

// ==================== BOOKTABS TABLE HELPER ====================
#let booktable(columns: auto, align: auto, header: (), rows: (), font-size: 8.1pt, inset: (x: 4.5pt, y: 3.4pt), row-fills: (:)) = {
  set text(size: font-size)
  set par(justify: false, leading: 0.45em)
  table(
    columns: columns,
    align: align,
    inset: inset,
    stroke: (x, y) => (
      top: if y == 0 { 1.2pt + navy } else if y == 1 { 0.65pt + navy } else { none },
      bottom: 0.3pt + rgb("#E2E8F0"),
      left: none,
      right: none,
    ),
    fill: (x, y) => {
      if y == 0 { head-bg }
      else if str(y) in row-fills { row-fills.at(str(y)) }
      else if calc.odd(y) { white }
      else { rgb("#FAFBFD") }
    },
    table.header(..header.map(h => text(weight: "bold", fill: navy)[#h])),
    ..rows.flatten(),
    table.hline(stroke: 1.2pt + navy),
  )
}

// Helper for Appendix numbering (A, B, C...)
#let start-appendix() = {
  counter(heading).update(0)
  set heading(numbering: "A.1")
}
