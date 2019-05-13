var seed = 17;
function random() {
    var x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

function generateGraph(svg, graph, linklength, strength) {

  var width = 300;
  var height = 200;

  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) { return d.id; })
                       .strength(2)
                       .distance(function() {return linklength}))
      .force("charge", d3.forceManyBody().strength(-110))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x",d3.forceX(1000).strength(strength))
      .force("y",d3.forceY(1000).strength(strength))
      .stop()
  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", 1)
      .attr("stroke","black")

  var node = svg.append("g")
      .attr("class", "nodes")
    

  var nodedata = node
      .selectAll("circle")
      .data(graph.nodes)
      .enter().append("circle")
      .attr("r", 3)
      .attr("fill", "black")

  nodedata.append("title")
      .text(function(d) { return d.id; });

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);

  simulation.force("link")
      .links(graph.links);

  for (var i = 0; i < 200; ++i) {
    simulation.tick();
  }

  function ticked() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    nodedata.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }

  var updateColors = function (update) {
    node.selectAll("circle")
        .data(update)
        .attr("fill", function(d,i) { return d})
  }

  ticked()

  return updateColors
};


function genGrid(svg) {

  var data = {"nodes": [],"links": []}
  var id = {}

  var n = 6
  var c = 0
  for (var i = 0; i < n; i ++) {
    for (var j = 0; j < n; j ++) {
      data.nodes.push({"id": i + "," + j})
      id[i + "," + j] = c
      c++
    } 
  }

  var L = zeros2D(c,c)

  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++){
      if (j < (n-1)) {
        data.links.push({"source":i + "," + j, "target": i + "," + (j + 1)})
        L[id[i + "," + j]][id[i + "," + (j+1)]] = -1
        L[id[i + "," + (j+1)]][id[i + "," + j]] = -1
      }
      if (i < (n-1)) {
        data.links.push({"source":i + "," + j, "target": (i + 1) + "," + j})
        L[id[i + "," + j]][id[(i+1) + "," + j]] = -1
        L[id[(i+1) + "," + j]][id[i + "," + j]] = -1
      }
    }
  }

  for (var i = 0; i < c; i++) {
    L[i][i] = -numeric.sum(L[i]) + ((i == 0) ? 1 : 0)
  }
  
  colorbrewer.Spectral[3]
  var divergent = d3.scaleLinear().domain([-0.1,0,0.1]).range(["#fc8d59", "#e6e600", "#99d594"]);

  var E = eigSym(L)

  var update = generateGraph(svg, data,2,0.2)

  var colors = []
  for (var i = 0; i < c; i++) {
    colors.push(divergent(E.U[34][i]))
  }
  update(colors)


}


function genPath(svg) {
  var data = {"nodes": [], "links": []}

  var n = 36
  var L = zeros2D(n,n)
  for (var i = 0; i < n; i ++) {
    data.nodes.push({"id": i})
    if (i < (n-1)){ 
      data.links.push({"source":i, "target": i+1})    
      L[i][i+1] = -1
      L[i+1][i] = -1 
    }   
  }

  for (var i = 0; i < 3; i++){
    var s = Math.floor(random()*36)
    var t = Math.floor(random()*36)
    data.links.push({"source":s, "target": t})
    L[s][t] = -1
    L[t][s] = -1
  }

  for (var i = 0; i < n; i++) {
    L[i][i] = -numeric.sum(L[i]) + ((i == 0) ? 1 : 0)
  }

  var divergent = d3.scaleLinear().domain([-0.1,0,0.1]).range(["#fc8d59", "#e6e600", "#99d594"]);

  var E = eigSym(L)

  var update = generateGraph(svg, data, 3, 0.43)
  var colors = []
  for (var i = 0; i < n; i++) {
    colors.push(divergent(E.U[34][i]))
  }
  update(colors)


}


function genExpander(svg) {
  var data = {"nodes": [], "links": []}
  var n = 36
  var L = zeros2D(n,n)

  for (var i = 0; i < n; i ++) {
    data.nodes.push({"id": i})   
  }

  for (var i = 0; i < 80; i++){
    var s = Math.floor(random()*36);
    var t = Math.floor(random()*36)
    data.links.push({"source":s, "target": t})
    L[s][t] = -1
    L[t][s] = -1
  }

  for (var i = 0; i < n; i++) {
    L[i][i] = -numeric.sum(L[i]) + ((i == 0) ? 1 : 0)
  }
  var divergent = d3.scaleLinear().domain([-0.1,0,0.1]).range(["#fc8d59", "#e6e600", "#99d594"]);

  var E = eigSym(L)
  var update =  generateGraph(svg, data, 20, 0.2)
  var colors = []
  for (var i = 0; i < n; i++) {
    colors.push(divergent(E.U[34][i]))
  }
  update(colors)
}

