Bun.serve({
  fetch(req) {
    const url = new URL(req.url);

    if (url.pathname.endsWith("/") || url.pathname.endsWith("/index.html"))
      return new Response(Bun.file(import.meta.dir + "/index.html"));

    if (url.pathname.endsWith("/tfjs_model/model.json"))
      return new Response(Bun.file(import.meta.dir + "/tfjs_model/model.json"));

    if (url.pathname.endsWith("/tfjs_model/group1-shard1of1.bin"))
      return new Response(Bun.file(import.meta.dir + "/tfjs_model/group1-shard1of1.bin"));

    if (url.pathname.endsWith("/tfjs_model/assets.zip"))
      return new Response(Bun.file(import.meta.dir + "/tfjs_model/assets.zip"));

    // all other routes
    return new Response("Hello!");
  },
});
