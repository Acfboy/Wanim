// CodeMirror 6 editor for the playground.
//
// This file is *source*: scripts/build-editor.mjs bundles it into
// dist/editor.browser.js, and demo.js loads that bundle (the page never
// imports this file directly).  The setup mirrors the website playground
// (React CodeMirrorEditor): Python language support, the One Dark theme and
// the same basicSetup options.
import { EditorView, keymap } from "@codemirror/view";
import { indentWithTab } from "@codemirror/commands";
import { basicSetup } from "@uiw/codemirror-extensions-basic-setup";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

/**
 * Mount a Python editor inside `parent`.
 *
 * Returns `{ view, getValue, setValue }`: `view` is the raw EditorView, the
 * other two are the small surface the demo page needs.
 */
export function createEditor({ parent, doc, onChange, ariaLabel = "Python source" }) {
  const view = new EditorView({
    parent,
    doc,
    extensions: [
      basicSetup({
        lineNumbers: true,
        foldGutter: false,
        highlightActiveLine: true,
        highlightActiveLineGutter: true,
        autocompletion: true,
        bracketMatching: true,
        closeBrackets: true,
        indentOnInput: true,
        tabSize: 4,
      }),
      python(),
      oneDark,
      // @uiw/react-codemirror (the website playground) enables Tab indentation
      // by default; keep the same behaviour here.
      keymap.of([indentWithTab]),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) onChange?.(update.state.doc.toString());
      }),
      EditorView.contentAttributes.of({ "aria-label": ariaLabel }),
    ],
  });
  return {
    view,
    getValue() {
      return view.state.doc.toString();
    },
    setValue(value) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
    },
  };
}
