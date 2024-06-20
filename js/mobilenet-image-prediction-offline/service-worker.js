var CACHE_NAME = "mobilenet-image-prediction-offline"; // edit this file to trigger PWA update when the site is refreshed in Chrome

var offlinePage = "/index.html";

var URLS = [
  // yes, the slash "/" matters here in service-worker.js:
  offlinePage,
  "/service-worker.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0",
  "https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0",
  // "model.json",
  // "group1-shard1of5.bin",
  // "group1-shard2of5.bin",
  // "group1-shard3of5.bin",
  // "group1-shard4of5.bin",
  // "group1-shard5of5.bin",
  "https://www.kaggle.com/models/google/mobilenet-v1/tfJs/100-224-classification/1/model.json?tfjs-format=file&tfhub-redirect=true",
  // "https://storage.googleapis.com/kagglesdsdata/models/1586/1951/model.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240620%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240620T025657Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b9d7bd22cfe3a20c629b2cf16c23d2456db9901617689c44d8a70af628aebe7f5bfff23b047503df86f1e3980b1cd1703c1ac8e1b473645bdbcf01c6841b7ca7afa49e3faf3f5ee9adcade2dfcf854529e9306fcfc180f0cd5bde79e4d82eceb171004dc110c5e660d4a47114c86704a41e2bda2601b65ee67b2bfdfef81d5962d28dd87d7970b7ddae6f9f777846662834a4f885409f388721cfd7459ff4d3fc7548ec1142217bf98b65e6d190b2ee0838ca4e5cd3f1b3c6f5cc822899259407653f3f2c528e9a4947e50cee64f26c3f110d9db128790695956b7f686dae232e109cc52c41cc37caceca82bae8b7f5c57652fa036b3b0858c9dabf94cd23155",
  // "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1/model.json?tfjs-format=file",
];

self.addEventListener("install", installServiceWorker);
self.addEventListener("activate", activateServiceWorker);
self.addEventListener("fetch", interceptResourceFetchWithServiceWorker);

function installServiceWorker(event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("Service worker installing.");
      return cache.addAll(URLS);
    })
  );
}

function activateServiceWorker(event) {
  event.waitUntil(
    caches
      .keys() // cache names (caches)
      .then((cacheKeys) => {
        // cache entries (keys/entries in a single cache)
        const oldKeys = cacheKeys.filter(
          (key) => key.indexOf(CACHE_NAME) !== 0
        );
        // promise to delete all old keys in this cache:
        const promisesToDeleteOldKeys = oldKeys.map((oldKey) =>
          caches.delete(oldKey)
        );
        // don't continue until ALL old keys are deleted:
        return Promise.all(promisesToDeleteOldKeys);
      })
  );
}

function interceptResourceFetchWithServiceWorker(event) {
  var url = new URL(event.request.url);
  if (URLS.indexOf(url.pathname) !== -1) {
    event.respondWith(
      caches
        .match(event.request)
        .then(function (response) {
          if (!response) {
            throw new Error(event.request + " not found in cache");
          }
          console.log("Service worker working even though you are offline.");
          return response;
        })
        .catch(function (error) {
          fetch(event.request);
        })
    );
  } else if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request).catch(function (error) {
        return caches.open(CACHE_NAME).then(function (cache) {
          console.log("Service worker working even though you are offline.");
          // return cache.matchAll(URLS);
          return cache.match("index.html"); // or offline.html
        });
      })
    );
  }
}
