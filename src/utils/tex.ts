import { readFileSync } from "node:fs";

export const DEFAULT_PREAMBLE =
  "\\usepackage[english]{babel}\n\\usepackage{amsmath}\n\\usepackage{amssymb}";

export const BEGIN_DOCUMENT = "\\begin{document}";
export const END_DOCUMENT = "\\end{document}";

export interface TexTemplateOptions {
  tex_compiler?: string | string[];
  description?: string;
  output_format?: string;
  documentclass?: string;
  preamble?: string;
  placeholder_text?: string;
  post_doc_commands?: string;
  body?: string;
}

export class TexTemplate {
  private _body = "";

  tex_compiler: string | string[];
  description: string;
  output_format: string;
  documentclass: string;
  preamble: string;
  placeholder_text: string;
  post_doc_commands: string;

  constructor(options: TexTemplateOptions = {}) {
    this.tex_compiler = options.tex_compiler ?? "latex";
    this.description = options.description ?? "";
    this.output_format = options.output_format ?? ".dvi";
    this.documentclass = options.documentclass ?? "\\documentclass[preview]{standalone}";
    this.preamble = options.preamble ?? DEFAULT_PREAMBLE;
    this.placeholder_text = options.placeholder_text ?? "YourTextHere";
    this.post_doc_commands = options.post_doc_commands ?? "";
    if (options.body !== undefined) this._body = options.body;
  }

  get body(): string {
    return (
      this._body ||
      [this.documentclass, this.preamble, BEGIN_DOCUMENT, this.post_doc_commands, this.placeholder_text, END_DOCUMENT]
        .filter((part) => part)
        .join("\n")
    );
  }

  set body(value: string) {
    this._body = value;
  }

  static from_file(file = "tex_template.tex", options: TexTemplateOptions = {}): TexTemplate {
    const instance = new TexTemplate(options);
    instance.body = readFileSync(file, { encoding: "utf-8" });
    return instance;
  }

  static fromFile(file = "tex_template.tex", options: TexTemplateOptions = {}): TexTemplate {
    return TexTemplate.from_file(file, options);
  }

  add_to_preamble(txt: string, prepend = false): this {
    if (this._body) {
      console.warn(
        "This TeX template was created with a fixed body, trying to add text the preamble will have no effect.",
      );
    }
    if (prepend) this.preamble = `${txt}\n${this.preamble}`;
    else this.preamble += `\n${txt}`;
    return this;
  }

  addToPreamble(txt: string, prepend = false): this {
    return this.add_to_preamble(txt, prepend);
  }

  add_to_document(txt: string): this {
    if (this._body) {
      console.warn(
        "This TeX template was created with a fixed body, trying to add text the document will have no effect.",
      );
    }
    this.post_doc_commands += txt;
    return this;
  }

  addToDocument(txt: string): this {
    return this.add_to_document(txt);
  }

  get_texcode_for_expression(expression: string): string {
    return this.body.replace(this.placeholder_text, expression);
  }

  getTexcodeForExpression(expression: string): string {
    return this.get_texcode_for_expression(expression);
  }

  get_texcode_for_expression_in_env(expression: string, environment: string): string {
    const [begin, end] = _texcode_for_environment(environment);
    return this.body.replace(this.placeholder_text, [begin, expression, end].join("\n"));
  }

  getTexcodeForExpressionInEnv(expression: string, environment: string): string {
    return this.get_texcode_for_expression_in_env(expression, environment);
  }

  copy(): this {
    const clone = new TexTemplate({
      tex_compiler: Array.isArray(this.tex_compiler) ? [...this.tex_compiler] : this.tex_compiler,
      description: this.description,
      output_format: this.output_format,
      documentclass: this.documentclass,
      preamble: this.preamble,
      placeholder_text: this.placeholder_text,
      post_doc_commands: this.post_doc_commands,
    });
    clone._body = this._body;
    return clone as this;
  }
}

export function _texcode_for_environment(environment: string): [string, string] {
  const withoutBegin = removePrefix(environment, "\\begin");
  const stripped = removePrefix(withoutBegin, "{");

  // The \begin command takes everything and closes with a brace.
  let begin = `\\begin{${stripped}`;
  // If it doesn't end on } or ], assume missing }.
  if (!begin.endsWith("}") && !begin.endsWith("]")) begin += "}";

  // While the \end command terminates at the first closing brace.
  const splitAtBrace = stripped.split("}", 2);
  const end = `\\end{${splitAtBrace[0] ?? ""}}`;

  return [begin, end];
}

function removePrefix(value: string, prefix: string): string {
  return value.startsWith(prefix) ? value.slice(prefix.length) : value;
}
