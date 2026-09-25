import { describe, expect, it, vi, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { TexTemplate, _texcode_for_environment } from "../src/utils/tex.js";
import {
  convert_to_svg,
  make_tex_compilation_command,
  print_tex_error,
} from "../src/utils/tex_file_writing.js";

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// manim.utils.tex.TexTemplate.body
// Python ground truth (verbatim interpreter output):
//   '\documentclass[preview]{standalone}\n\usepackage[english]{babel}\n
//    \usepackage{amsmath}\n\usepackage{amssymb}\n\begin{document}\n
//    YourTextHere\n\end{document}'
// ---------------------------------------------------------------------------
describe("TexTemplate.body", () => {
  it("builds the default body from documentclass/preamble/document", () => {
    expect(new TexTemplate().body).toBe(
      "\\documentclass[preview]{standalone}\n" +
        "\\usepackage[english]{babel}\n" +
        "\\usepackage{amsmath}\n" +
        "\\usepackage{amssymb}\n" +
        "\\begin{document}\n" +
        "YourTextHere\n" +
        "\\end{document}",
    );
  });
  it("returns a fixed body once set", () => {
    const t = new TexTemplate();
    t.body = "CUSTOM BODY";
    expect(t.body).toBe("CUSTOM BODY");
  });
  it("omits empty post_doc_commands without leaving blank lines", () => {
    // No "\n\n" may appear between \begin{document} and the placeholder.
    expect(new TexTemplate().body).not.toContain("\n\n");
  });
  it("reads the body from a file with from_file", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-tex-"));
    try {
      const file = path.join(dir, "tex_template.tex");
      writeFileSync(file, "FILE BODY", { encoding: "utf-8" });
      expect(TexTemplate.from_file(file).body).toBe("FILE BODY");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("warns when adding to the preamble of a fixed-body template", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const t = new TexTemplate();
    t.body = "FIXED";
    t.add_to_preamble("\\usepackage{foo}");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(t.preamble).toContain("\\usepackage{foo}");
  });
  it("copies independently", () => {
    const t = new TexTemplate();
    const c = t.copy();
    c.preamble += "\n\\usepackage{foo}";
    expect(t.preamble).not.toContain("foo");
    expect(c.get_texcode_for_expression("X")).toContain("X");
  });
});

// ---------------------------------------------------------------------------
// manim.utils.tex._texcode_for_environment
// Python ground truths (verbatim interpreter output):
//   'align*' -> ('\begin{align*}', '\end{align*}')
//   '{tabular}[t]{cccl}' -> ('\begin{tabular}[t]{cccl}', '\end{tabular}')
//   'tabular}{cccl' -> ('\begin{tabular}{cccl}', '\end{tabular}')
//   '\begin{tabular}[t]{cccl}' -> ('\begin{tabular}[t]{cccl}', '\end{tabular}')
// ---------------------------------------------------------------------------
describe("_texcode_for_environment", () => {
  it("wraps a bare environment name", () => {
    expect(_texcode_for_environment("align*")).toEqual([
      "\\begin{align*}",
      "\\end{align*}",
    ]);
  });
  it("keeps extra args on \\begin but drops them from \\end", () => {
    expect(_texcode_for_environment("{tabular}[t]{cccl}")).toEqual([
      "\\begin{tabular}[t]{cccl}",
      "\\end{tabular}",
    ]);
  });
  it("repairs a missing closing brace on \\begin", () => {
    expect(_texcode_for_environment("tabular}{cccl}")).toEqual([
      "\\begin{tabular}{cccl}",
      "\\end{tabular}",
    ]);
  });
  it("strips a leading \\begin", () => {
    expect(_texcode_for_environment("\\begin{tabular}[t]{cccl}")).toEqual([
      "\\begin{tabular}[t]{cccl}",
      "\\end{tabular}",
    ]);
  });
});

// ---------------------------------------------------------------------------
// manim.utils.tex_file_writing.make_tex_compilation_command
// Python ground truth (from source): latex/pdflatex/luatex/lualatex share one
// shape; xelatex has -no-pdf only for .xdv; anything else raises.
// ---------------------------------------------------------------------------
describe("make_tex_compilation_command", () => {
  it("builds the latex command", () => {
    expect(make_tex_compilation_command("latex", ".dvi", "f.tex", "d")).toEqual([
      "latex",
      "-interaction=batchmode",
      "-output-format=dvi",
      "-halt-on-error",
      "-output-directory=d",
      "f.tex",
    ]);
  });
  it("passes -no-pdf for xelatex xdv output", () => {
    expect(make_tex_compilation_command("xelatex", ".xdv", "f.tex", "d")).toEqual([
      "xelatex",
      "-no-pdf",
      "-interaction=batchmode",
      "-halt-on-error",
      "-output-directory=d",
      "f.tex",
    ]);
  });
  it("omits -no-pdf for xelatex pdf output", () => {
    const cmd = make_tex_compilation_command("xelatex", ".pdf", "f.tex", "d");
    expect(cmd[0]).toBe("xelatex");
    expect(cmd).not.toContain("-no-pdf");
  });
  it("rejects unknown compilers and bad xelatex formats", () => {
    expect(() => make_tex_compilation_command("unknown", ".dvi", "f.tex", "d")).toThrow(
      "Tex compiler unknown unknown.",
    );
    expect(() => make_tex_compilation_command("xelatex", ".dvi", "f.tex", "d")).toThrow(
      "xelatex output is either pdf or xdv",
    );
  });
});

// ---------------------------------------------------------------------------
// manim.utils.tex_file_writing.convert_to_svg
// Python: returns the .svg path; if missing, runs dvisvgm (with --pdf for
// .pdf input) and raises ValueError naming dvisvgm when still absent.
// ---------------------------------------------------------------------------
describe("convert_to_svg", () => {
  it("returns the existing svg path without converting", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-svg-"));
    try {
      const dvi = path.join(dir, "out.dvi");
      const svg = path.join(dir, "out.svg");
      writeFileSync(dvi, "dvi", { encoding: "utf-8" });
      writeFileSync(svg, "<svg/>", { encoding: "utf-8" });
      expect(convert_to_svg(dvi, ".dvi")).toBe(svg);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("raises a dvisvgm error when conversion produces nothing", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "wanim-svg-"));
    try {
      const dvi = path.join(dir, "missing.dvi");
      writeFileSync(dvi, "dvi", { encoding: "utf-8" });
      expect(() => convert_to_svg(dvi, ".dvi")).toThrow(
        "does not support converting",
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// manim.utils.tex_file_writing.print_tex_error
// Python: logs 'LaTeX compilation error: <msg>'; prints a 'Context of error'
// block marking the failing line with '-> '; returns None when the failing
// line lies outside the user source.
// ---------------------------------------------------------------------------
describe("print_tex_error", () => {
  it("logs the error and marks the failing line", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    const log = [
      "! LaTeX Error: File `foo.sty' not found.\n",
      "l.5 \\usepackage{foo}\n",
    ];
    const source = [
      "line1\n",
      "line2\n",
      "line3\n",
      "line4\n",
      "line5 \\usepackage{foo}\n",
      "line6\n",
    ];
    print_tex_error(log, 0, source);
    expect(String(err.mock.calls[0]![0])).toContain(
      "LaTeX compilation error: LaTeX Error: File `foo.sty' not found.",
    );
    const context = String(err.mock.calls[1]![0]);
    expect(context).toContain("Context of error");
    expect(context).toContain("-> ");
    // The package insight fires for a missing-file error.
    expect(info.mock.calls.some((call) => String(call[0]).includes("foo.sty"))).toBe(
      true,
    );
  });
  it("returns null when the error line is outside the source", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(print_tex_error(["! boom\n", "l.100 x\n"], 0, ["a\n"])).toBeNull();
    expect(err).toHaveBeenCalledTimes(1);
  });
});
