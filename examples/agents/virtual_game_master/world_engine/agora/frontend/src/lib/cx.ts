/** Merge class names, filtering out falsy values */
export function cx(...args: (string | false | null | undefined)[]): string {
  return args.filter(Boolean).join(' ');
}
