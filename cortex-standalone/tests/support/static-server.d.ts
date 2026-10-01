export interface StaticServer {
  port: number;
  url: string;
  origin: string;
  requests: Array<{ method: string; path: string }>;
  mount(prefix: string, dir: string | null): void;
  unmount(prefix: string): void;
  stop(): Promise<void>;
}
export function start(opts: { root: string; mounts?: Record<string, string | null>; port?: number }): Promise<StaticServer>;
export const MIME: Record<string, string>;
