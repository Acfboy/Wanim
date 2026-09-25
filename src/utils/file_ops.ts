const templateNames = ["Axes", "Default", "MovingCamera"];

function templateUrl(): string {
  return decodeURIComponent(new URL("../../src/templates", import.meta.url).pathname).replace(/\/$/, "");
}

export function get_template_names(): string[] {
  return [...templateNames];
}

export function get_template_path(): string {
  return templateUrl();
}
