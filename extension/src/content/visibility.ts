export interface VisibilityState {
  visible: boolean;
  occluded: boolean;
  inViewport: boolean;
  bbox: number[];
}

const HIDDEN_TAGS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS", "HEAD", "META",
  "LINK", "TITLE", "IFRAME", "OBJECT", "EMBED", "AUDIO", "VIDEO",
]);

export function isAriaHidden(element: Element): boolean {
  let node: Element | null = element;
  while (node) {
    if (node.getAttribute("aria-hidden") === "true") return true;
    node = node.parentElement;
  }
  return false;
}

export function hiddenReason(element: Element): string | null {
  if (!(element instanceof HTMLElement) && !(element instanceof SVGElement)) {
    return "not-renderable";
  }
  if (element instanceof HTMLElement && element.hidden) return "hidden";
  if (isAriaHidden(element)) return "aria-hidden";
  const style = getComputedStyle(element);
  if (style.display === "none") return "display-none";
  if (style.visibility === "hidden" || style.visibility === "collapse") {
    return "visibility-hidden";
  }
  if (Number.parseFloat(style.opacity || "1") <= 0) return "opacity-zero";
  if (Number.parseFloat(style.fontSize || "0") <= 0) return "font-size-zero";
  if (
    typeof element.checkVisibility === "function" &&
    !element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })
  ) {
    return "check-visibility";
  }
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return "zero-size";
  const positioned = ["absolute", "fixed", "sticky"].includes(style.position);
  if (
    positioned &&
    (
      rect.right < 0 ||
      rect.bottom < 0 ||
      rect.left > innerWidth + 2_000 ||
      rect.top > document.documentElement.scrollHeight + 1_000
    )
  ) {
    return "off-screen";
  }
  if (hasCamouflagedText(element, style)) return "matching-colours";
  return null;
}

export function isRendered(element: Element): boolean {
  return hiddenReason(element) === null;
}

export function visibilityState(element: Element): VisibilityState {
  const rect = element.getBoundingClientRect();
  const bbox = [rect.x, rect.y, rect.width, rect.height].map((n) =>
    Math.round(n * 10) / 10
  );
  const visible = isRendered(element);
  const inViewport =
    visible &&
    rect.bottom >= 0 &&
    rect.right >= 0 &&
    rect.top <= innerHeight &&
    rect.left <= innerWidth;
  let occluded = false;
  if (visible && inViewport) {
    const x = Math.max(0, Math.min(innerWidth - 1, rect.left + rect.width / 2));
    const y = Math.max(0, Math.min(innerHeight - 1, rect.top + rect.height / 2));
    const top = document.elementFromPoint(x, y);
    occluded = top !== element && !(top && element.contains(top));
  }
  return { visible, occluded, inViewport, bbox };
}

export function hiddenTag(element: Element): boolean {
  return HIDDEN_TAGS.has(element.tagName.toUpperCase());
}

function hasCamouflagedText(element: Element, style: CSSStyleDeclaration): boolean {
  if (!(element.textContent ?? "").trim()) return false;
  const foreground = rgba(style.color);
  const background = effectiveBackground(element);
  if (!foreground || !background || foreground[3]! <= 0) return false;
  return foreground.slice(0, 3).every(
    (component, index) => Math.abs(component - background[index]!) <= 2,
  );
}

function effectiveBackground(element: Element): number[] | null {
  let current: Element | null = element;
  while (current) {
    const parsed = rgba(getComputedStyle(current).backgroundColor);
    if (parsed && parsed[3]! > 0.01) return parsed;
    current = current.parentElement;
  }
  const fallback = rgba(
    getComputedStyle(document.body ?? document.documentElement).backgroundColor,
  );
  return fallback && fallback[3]! > 0.01 ? fallback : [255, 255, 255, 1];
}

function rgba(value: string): number[] | null {
  const match = /rgba?\(([^)]+)\)/i.exec(value);
  if (!match?.[1]) return null;
  const values = match[1].split(/[\s,/]+/).filter(Boolean).map(Number);
  if (values.length < 3 || values.some((item) => !Number.isFinite(item))) return null;
  return [values[0]!, values[1]!, values[2]!, values[3] ?? 1];
}
