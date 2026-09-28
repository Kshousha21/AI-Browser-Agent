import { isProductUrl } from "./product-scope";
/** Compatibility argument is ignored; Chrome activeTab/site grants authorize injection. */
export function isAllowed(url: string, _legacyPatterns?: readonly string[]): boolean {
  return isProductUrl(url);
}
