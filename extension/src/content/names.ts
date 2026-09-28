import { hiddenTag, isAriaHidden } from "./visibility";

const IMPLICIT_ROLES: Record<string, string> = {
  A: "link", AREA: "link", ARTICLE: "article", ASIDE: "complementary",
  BUTTON: "button", DETAILS: "group", DIALOG: "dialog", FIELDSET: "group",
  FIGURE: "figure", FOOTER: "contentinfo", FORM: "form", H1: "heading",
  H2: "heading", H3: "heading", H4: "heading", H5: "heading", H6: "heading",
  HEADER: "banner", HR: "separator", IMG: "img", LI: "listitem", MAIN: "main",
  NAV: "navigation", OL: "list", OPTION: "option", OUTPUT: "status",
  PROGRESS: "progressbar", SECTION: "region", SELECT: "combobox",
  SUMMARY: "button", TABLE: "table", TD: "cell", TEXTAREA: "textbox",
  TH: "columnheader", TR: "row", UL: "list",
};

const INPUT_ROLES: Record<string, string> = {
  button: "button", checkbox: "checkbox", email: "textbox", image: "button",
  number: "spinbutton", radio: "radio", range: "slider", reset: "button",
  search: "searchbox", submit: "button", tel: "textbox", text: "textbox",
  url: "textbox",
};

const NAME_FROM_CONTENT = new Set([
  "button", "cell", "checkbox", "columnheader", "gridcell", "heading", "link",
  "menuitem", "menuitemcheckbox", "menuitemradio", "option", "radio", "row",
  "rowheader", "switch", "tab", "tooltip", "treeitem",
]);

const LABELABLE = new Set([
  "BUTTON", "INPUT", "METER", "OUTPUT", "PROGRESS", "SELECT", "TEXTAREA",
]);

export function normalizeText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export function computeRole(element: Element): string {
  const explicit = element.getAttribute("role")?.trim().split(/\s+/)[0];
  if (explicit) return explicit;
  const tag = element.tagName.toUpperCase();
  if (tag === "A" || tag === "AREA") {
    return element.hasAttribute("href") ? "link" : "generic";
  }
  if (tag === "INPUT") {
    const type = (element as HTMLInputElement).type.toLowerCase();
    if (type === "hidden") return "none";
    if (type === "text" && element.hasAttribute("list")) return "combobox";
    return INPUT_ROLES[type] ?? "textbox";
  }
  if (tag === "SELECT") {
    const select = element as HTMLSelectElement;
    return select.multiple || select.size > 1 ? "listbox" : "combobox";
  }
  return IMPLICIT_ROLES[tag] ?? "generic";
}

function labelsFor(element: Element): HTMLLabelElement[] {
  if (!LABELABLE.has(element.tagName.toUpperCase())) return [];
  const native = (element as HTMLInputElement).labels;
  if (native) return Array.from(native);
  const labels: HTMLLabelElement[] = [];
  if (element.id) {
    const root = element.getRootNode() as Document | ShadowRoot;
    const escaped = CSS.escape(element.id);
    labels.push(
      ...Array.from(root.querySelectorAll<HTMLLabelElement>(`label[for="${escaped}"]`)),
    );
  }
  const ancestor = element.closest("label");
  if (ancestor instanceof HTMLLabelElement && !labels.includes(ancestor)) {
    labels.push(ancestor);
  }
  return labels;
}

interface NameContext {
  visited: Set<Element>;
  inContent: boolean;
  labelledByTarget: boolean;
}

function childText(element: Element, ctx: NameContext): string {
  const parts: string[] = [];
  for (const child of Array.from(element.childNodes)) {
    if (child.nodeType === Node.TEXT_NODE) {
      parts.push(child.textContent ?? "");
    } else if (child.nodeType === Node.ELEMENT_NODE) {
      const item = child as Element;
      if (hiddenTag(item)) continue;
      parts.push(
        accessibleName(item, {
          visited: ctx.visited,
          inContent: true,
          labelledByTarget: false,
        }),
      );
    }
  }
  return normalizeText(parts.join(" "));
}

/** A compact implementation of the accname ordering used by Chrome. */
export function accessibleName(
  element: Element,
  ctx: NameContext = {
    visited: new Set(),
    inContent: false,
    labelledByTarget: false,
  },
): string {
  if (ctx.visited.has(element)) return "";
  ctx.visited.add(element);
  if (!ctx.labelledByTarget && isAriaHidden(element)) return "";

  const labelledBy = element.getAttribute("aria-labelledby");
  if (labelledBy && !ctx.inContent && !ctx.labelledByTarget) {
    const root = element.getRootNode() as Document | ShadowRoot;
    const text = normalizeText(
      labelledBy
        .split(/\s+/)
        .filter(Boolean)
        .map((id) => {
          const target = root.getElementById?.(id) ?? document.getElementById(id);
          if (!target) return "";
          return accessibleName(target, {
            visited: ctx.visited,
            inContent: false,
            labelledByTarget: true,
          }) || normalizeText(target.textContent ?? "");
        })
        .join(" "),
    );
    if (text) return text;
  }

  const ariaLabel = element.getAttribute("aria-label");
  if (ariaLabel && normalizeText(ariaLabel)) return normalizeText(ariaLabel);

  const tag = element.tagName.toUpperCase();
  if (tag === "INPUT") {
    const input = element as HTMLInputElement;
    const type = input.type.toLowerCase();
    if (["button", "submit", "reset"].includes(type)) {
      return normalizeText(input.value) ||
        (type === "submit" ? "Submit" : type === "reset" ? "Reset" : "");
    }
    if (type === "image") return normalizeText(input.alt) || "Submit";
  }
  if (tag === "IMG" || tag === "AREA") {
    const alt = element.getAttribute("alt");
    if (alt !== null) return normalizeText(alt);
  }
  if (tag === "FIELDSET") {
    const caption = element.querySelector("legend");
    if (caption) return normalizeText(caption.textContent ?? "");
  }
  if (tag === "TABLE") {
    const caption = element.querySelector("caption");
    if (caption) return normalizeText(caption.textContent ?? "");
  }
  if (tag === "FIGURE") {
    const caption = element.querySelector("figcaption");
    if (caption) return normalizeText(caption.textContent ?? "");
  }

  const labels = labelsFor(element);
  if (labels.length) {
    const text = normalizeText(labels.map((label) => label.textContent ?? "").join(" "));
    if (text) return text;
  }
  const role = computeRole(element);
  if (ctx.inContent || NAME_FROM_CONTENT.has(role)) {
    const text = childText(element, ctx);
    if (text) return text;
  }
  const title = normalizeText(element.getAttribute("title") ?? "");
  if (title) return title;
  return normalizeText(element.getAttribute("placeholder") ?? "");
}
