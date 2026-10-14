import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const host = "127.0.0.1";
const port = Number(process.env.PANORAMA_GRAPH_PORT || 8765);
const moduleRoot = fileURLToPath(new URL("..", import.meta.url));
const dataRoot = path.join(moduleRoot, "results", "trajectory_sequences_180");
const apiKey = (process.env.GOOGLE_MAPS_API_KEY || "").trim();
const index = JSON.parse(fs.readFileSync(path.join(dataRoot, "index.json"), "utf8"));
const routes = new Map(index.map((route) => [route.id, route]));

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function safeJson(value) {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

function page(routeId) {
  const route = routes.get(routeId) || index[0];
  const manifest = JSON.parse(
    fs.readFileSync(path.join(dataRoot, route.id, "trajectory_180.json"), "utf8"),
  );
  const options = index.map((item) => (
    `<option value="${escapeHtml(item.id)}"${item.id === route.id ? " selected" : ""}>${escapeHtml(item.name)}</option>`
  )).join("");
  const frames = manifest.frames.map((frame) => {
    const number = String(frame.frame).padStart(2, "0");
    const heading = Number(frame.center_heading ?? frame.render_heading ?? frame.heading ?? 0).toFixed(1);
    return `<figure class="frame">
      <img src="/images/${encodeURIComponent(route.id)}/${encodeURIComponent(frame.file)}" alt="${escapeHtml(manifest.name)} frame ${number}" loading="lazy">
      <figcaption><strong>frame_${number}</strong><span>heading ${heading}&deg;</span></figcaption>
    </figure>`;
  }).join("");

  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Google Street View trajectories</title>
<style>
  *{box-sizing:border-box} html,body{margin:0;background:#f6f7f8;color:#202124;font:14px/1.45 Arial,"Microsoft YaHei",sans-serif}
  header{height:64px;background:#fff;border-bottom:1px solid #dfe1e5;display:flex;align-items:center;gap:24px;padding:0 24px;position:sticky;top:0;z-index:20}
  h1{font-size:18px;margin:0;white-space:nowrap} .summary{color:#5f6368;white-space:nowrap}
  label{margin-left:auto;color:#5f6368} select{margin-left:8px;width:min(330px,42vw);height:36px;border:1px solid #bdc1c6;border-radius:4px;background:#fff;padding:0 34px 0 10px;color:#202124}
  main{max-width:1500px;margin:auto;padding:24px}
  .section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:0 0 12px}
  h2{font-size:20px;margin:0} .meta{color:#5f6368;font-size:13px}
  #map,#fallback{height:min(58vh,620px);min-height:430px;width:100%;border:1px solid #dadce0;background:#e8eaed}
  #fallback{display:none;place-items:center;color:#5f6368;text-align:center;padding:24px}
  #status{min-height:22px;color:#5f6368;margin:9px 0 28px}
  .legend{display:flex;gap:18px;align-items:center}.legend span{display:inline-flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:50%;background:#d93025}.line{width:22px;height:3px;background:#d93025}
  .sequence{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
  .frame{margin:0;background:#fff;border:1px solid #dadce0;border-radius:6px;overflow:hidden}
  .frame img{display:block;width:100%;aspect-ratio:2/1;object-fit:cover;background:#e8eaed}
  figcaption{height:42px;display:flex;align-items:center;justify-content:space-between;padding:0 12px} figcaption span{color:#5f6368;font-size:12px}
  @media(max-width:760px){header{height:auto;min-height:64px;padding:12px 16px;flex-wrap:wrap;gap:6px 12px}.summary{display:none}label{width:100%;margin:0}select{width:calc(100% - 90px)}main{padding:16px}.sequence{grid-template-columns:1fr}#map,#fallback{height:55vh;min-height:360px}.section-head{align-items:start;flex-direction:column}}
</style>
</head>
<body>
<header>
  <h1>Google Street View trajectory viewer</h1>
  <span class="summary">100 条城市轨迹 &middot; 每条 8 帧 &middot; 水平 FOV 180&deg;</span>
  <label>选择 trajectory<select id="route-select">${options}</select></label>
</header>
<main>
  <div class="section-head"><div><h2>${escapeHtml(manifest.name)}</h2><div class="meta">Google Maps 底图 &middot; 采样点与 traversal path</div></div><div class="legend"><span><i class="line"></i>trajectory</span><span><i class="dot"></i>frame</span></div></div>
  <div id="map"></div><div id="fallback">Google Maps 暂时不可用。影像序列仍可正常浏览。</div>
  <div id="status">正在加载 Google Maps...</div>
  <div class="section-head"><div><h2>${escapeHtml(manifest.name)} &middot; image sequence</h2><div class="meta">真实 180&deg;投影 &middot; 1920&times;960</div></div></div>
  <section class="sequence">${frames}</section>
</main>
<script>
const trajectory=${safeJson(manifest)};
const apiKey=${safeJson(apiKey)};
const mapElement=document.getElementById("map");
const fallback=document.getElementById("fallback");
const statusElement=document.getElementById("status");

document.getElementById("route-select").addEventListener("change",(event)=>{
  location.href="/?route="+encodeURIComponent(event.target.value);
});

function openPano(frame){
  const heading=frame.center_heading??frame.render_heading??frame.heading??0;
  const url="https://www.google.com/maps/@?api=1&map_action=pano&pano="+encodeURIComponent(frame.pano_id)+"&heading="+encodeURIComponent(heading)+"&pitch=0&fov=90";
  window.open(url,"_blank","noopener");
}

function showFallback(message){
  mapElement.style.display="none";
  fallback.style.display="grid";
  statusElement.textContent=message;
}

window.initTrajectoryMap=()=>{
  try{
    const points=trajectory.frames.map((frame)=>({lat:Number(frame.lat),lng:Number(frame.lon)}));
    const bounds=new google.maps.LatLngBounds();
    points.forEach((point)=>bounds.extend(point));
    const map=new google.maps.Map(mapElement,{mapTypeId:"roadmap",mapTypeControl:true,streetViewControl:true,fullscreenControl:true,gestureHandling:"greedy"});
    new google.maps.Polyline({map,path:points,strokeColor:"#d93025",strokeOpacity:1,strokeWeight:4,zIndex:5});
    trajectory.frames.forEach((frame,index)=>{
      const name="frame_"+String(frame.frame).padStart(2,"0");
      const marker=new google.maps.Marker({
        map,
        position:points[index],
        title:name,
        label:{text:name,color:"#202124",fontSize:"11px",fontWeight:"700"},
        icon:{path:google.maps.SymbolPath.CIRCLE,scale:9,fillColor:index===0?"#f29900":"#ffffff",fillOpacity:1,strokeColor:"#d93025",strokeWeight:3},
        zIndex:10+index,
      });
      marker.addListener("click",()=>openPano(frame));
    });
    map.fitBounds(bounds,52);
    google.maps.event.addListenerOnce(map,"idle",()=>{
      if(map.getZoom()>20)map.setZoom(20);
    });
    statusElement.textContent="仅显示当前 trajectory；点击编号点可在 Google Maps 中打开对应 panorama。";
  }catch(error){showFallback("Google Maps 初始化失败："+error.message);}
};

if(!apiKey){
  showFallback("未配置 Google Maps API key。请使用 GOOGLE_MAPS_API_KEY 启动服务。");
}else{
  const script=document.createElement("script");
  script.src="https://maps.googleapis.com/maps/api/js?key="+encodeURIComponent(apiKey)+"&v=weekly&callback=initTrajectoryMap";
  script.async=true;
  script.defer=true;
  script.onerror=()=>showFallback("Google Maps API 加载失败。影像序列仍可正常浏览。");
  document.head.appendChild(script);
}
</script>
</body>
</html>`;
}

const server = http.createServer((request, response) => {
  try {
    const url = new URL(request.url, `http://${host}:${port}`);
    if (url.pathname === "/" || url.pathname === "/index.html") {
      const requested = url.searchParams.get("route");
      const routeId = routes.has(requested) ? requested : index[0].id;
      const body = page(routeId);
      response.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
      return response.end(body);
    }
    const match = /^\/images\/([a-z0-9_-]+)\/(frame_\d+\.jpg)$/i.exec(url.pathname);
    if (match && routes.has(match[1])) {
      const file = path.join(dataRoot, match[1], match[2]);
      if (fs.existsSync(file)) {
        response.writeHead(200, { "content-type": "image/jpeg", "cache-control": "public,max-age=3600" });
        return fs.createReadStream(file).pipe(response);
      }
    }
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  } catch (error) {
    response.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
    response.end(error.message);
  }
});

server.listen(port, host, () => {
  console.log(`Trajectory viewer: http://${host}:${port}/`);
});
