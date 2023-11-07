Bun.serve({
  fetch(req) {
    const url = new URL(req.url);

    if (url.pathname.endsWith("/") || url.pathname.endsWith("/index.html"))
      return new Response(Bun.file(import.meta.dir + "/index.html"));

    if (url.pathname.endsWith("/model.json"))
      return new Response(Bun.file(import.meta.dir + "/model.json"));

    // all other routes
    return new Response("Hello!");
  },
});
