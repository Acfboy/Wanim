import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import * as path from "node:path";

export type PathLike = string;

function asPosix(file: PathLike): string {
  return file;
}

function withSuffix(file: PathLike, suffix: string): string {
  const parsed = path.parse(file);
  return path.join(parsed.dir, `${parsed.name}${suffix}`);
}

export function make_tex_compilation_command(
  texCompiler: string,
  outputFormat: string,
  texFile: PathLike,
  texDir: PathLike,
): string[] {
  if (["latex", "pdflatex", "luatex", "lualatex"].includes(texCompiler)) {
    return [
      texCompiler,
      "-interaction=batchmode",
      `-output-format=${outputFormat.slice(1)}`,
      "-halt-on-error",
      `-output-directory=${asPosix(texDir)}`,
      `${asPosix(texFile)}`,
    ];
  } else if (texCompiler === "xelatex") {
    let outflag: string[];
    if (outputFormat === ".xdv") outflag = ["-no-pdf"];
    else if (outputFormat === ".pdf") outflag = [];
    else {
      const error = new Error("xelatex output is either pdf or xdv");
      error.name = "ValueError";
      throw error;
    }
    return [
      "xelatex",
      ...outflag,
      "-interaction=batchmode",
      "-halt-on-error",
      `-output-directory=${asPosix(texDir)}`,
      `${asPosix(texFile)}`,
    ];
  } else {
    const error = new Error(`Tex compiler ${texCompiler} unknown.`);
    error.name = "ValueError";
    throw error;
  }
}

export function makeTexCompilationCommand(
  texCompiler: string,
  outputFormat: string,
  texFile: PathLike,
  texDir: PathLike,
): string[] {
  return make_tex_compilation_command(texCompiler, outputFormat, texFile, texDir);
}

export function convert_to_svg(dviFile: PathLike, extension: string, page = 1): string {
  const result = withSuffix(dviFile, ".svg");
  if (!existsSync(result)) {
    const command = [
      "dvisvgm",
      ...(extension === ".pdf" ? ["--pdf"] : []),
      `--page=${page}`,
      "--no-fonts",
      "--verbosity=0",
      `--output=${asPosix(result)}`,
      `${asPosix(dviFile)}`,
    ];
    try {
      spawnSync(command[0]!, command.slice(1), { stdio: "ignore" });
    } catch {
      /* conversion failure is reported below */
    }
  }

  // If the file does not exist now, this means conversion failed.
  if (!existsSync(result)) {
    const error = new Error(
      `Your installation does not support converting ${path.parse(dviFile).ext} files to SVG.` +
        ` Consider updating dvisvgm to at least version 2.4.` +
        ` If this does not solve the problem, please refer to our troubleshooting guide at:` +
        ` https://docs.manim.community/en/stable/faq/general.html#my-installation-` +
        `does-not-support-converting-pdf-to-svg-help`,
    );
    error.name = "ValueError";
    throw error;
  }

  return result;
}

export function convertToSvg(dviFile: PathLike, extension: string, page = 1): string {
  return convert_to_svg(dviFile, extension, page);
}

export function* insight_inputenc_error(
  matching: RegExpMatchArray,
): Generator<string> {
  const codePoint = String.fromCodePoint(parseInt(matching[1]!, 16));
  // unicodedata.name() has no direct TS equivalent; report the code point.
  yield `TexTemplate does not support character '${codePoint}' (U+${matching[1]}).`;
  yield "See the documentation for manim.mobject.svg.tex_mobject for details on using a custom TexTemplate.";
}

export function* insight_package_not_found_error(
  matching: RegExpMatchArray,
): Generator<string> {
  yield `You do not have package ${matching[1]} installed.`;
  yield `Install ${matching[1]} it using your LaTeX package manager, or check for typos.`;
}

export const LATEX_ERROR_INSIGHTS: Array<{
  pattern: RegExp;
  getInsight: (matching: RegExpMatchArray) => Generator<string>;
}> = [
  {
    pattern: /inputenc Error: Unicode character (?:.*) \(U\+([0-9a-fA-F]+)\)/,
    getInsight: insight_inputenc_error,
  },
  {
    pattern: /LaTeX Error: File `(.*?[clsty])' not found/,
    getInsight: insight_package_not_found_error,
  },
];

export function print_tex_error(
  texCompilationLog: readonly string[],
  errorStartIndex: number,
  texSource: readonly string[],
): null | undefined {
  console.error(`LaTeX compilation error: ${texCompilationLog[errorStartIndex]!.slice(2)}`);

  // TeX errors eventually contain a line beginning 'l.xxx' where xxx is the
  // line number that caused the compilation failure. This code finds the next
  // such line after the current error message.
  const lineOfTexError =
    parseInt(
      texCompilationLog
        .slice(errorStartIndex)
        .filter((logLine) => logLine.startsWith("l."))[0]!
        .split(" ")[0]!
        .split(".")[1]!,
      10,
    ) - 1;
  // Our tex error may be on a line outside our user input because of post-processing.
  if (lineOfTexError >= texSource.length) return null;

  const context = ["Context of error: \n"];
  if (lineOfTexError < 3) {
    context.push(...texSource.slice(0, lineOfTexError + 3));
    context[context.length - 4] = `-> ${context[context.length - 4]}`;
  } else if (lineOfTexError > texSource.length - 3) {
    context.push(...texSource.slice(lineOfTexError - 1));
    context[1] = `-> ${context[1]}`;
  } else {
    context.push(...texSource.slice(lineOfTexError - 3, lineOfTexError + 3));
    context[context.length - 4] = `-> ${context[context.length - 4]}`;
  }

  console.error(context.join(""));

  for (const { pattern, getInsight } of LATEX_ERROR_INSIGHTS) {
    const matching = texCompilationLog[errorStartIndex]!.slice(2).match(pattern);
    if (matching !== null) {
      for (const insight of getInsight(matching)) console.info(insight);
    }
  }
  return undefined;
}

export function printTexError(
  texCompilationLog: readonly string[],
  errorStartIndex: number,
  texSource: readonly string[],
): null | undefined {
  return print_tex_error(texCompilationLog, errorStartIndex, texSource);
}
