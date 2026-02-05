import { useEffect, useState } from "react";

export function navigate(path: string) {
  if (window.location.pathname === path) return;
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export type Route =
  | { name: "home" }
  | { name: "cases_new" }
  | { name: "case_detail"; caseId: string };

function parseRoute(pathname: string): Route {
  if (pathname === "/" || pathname === "") return { name: "home" };
  if (pathname === "/cases/new") return { name: "cases_new" };
  const m = pathname.match(/^\/cases\/([^/]+)$/);
  if (m) return { name: "case_detail", caseId: decodeURIComponent(m[1]!) };
  return { name: "home" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.pathname));

  useEffect(() => {
    const onPop = () => setRoute(parseRoute(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return route;
}

