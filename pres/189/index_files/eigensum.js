
function renderEigenPanel(eigensum, U, x, b, wi, refit, hat, renderStars) {
  var mathdiv = eigensum.append("div")

  var xrange = d3.scaleLinear().domain([-10,10]).range([-1,1])
  var startpoint = 0
  var witemp = 0

  var equations = [];

  function renderEquation(hat, value, number) {
    var html = hat ? MathCache("phat") : MathCache("p");
    html = html.replace('<span class="mord mathrm">0</span>', value);
    html = html.replace('<span class="mord mathrm mtight">1</span>', number);
    return html;
  }

  if (!(renderStars === undefined)) {
    for (var i = 0; i < 7; i++) {

      mathdiv
        .append("span")
        .style("text-align","center")
        .style("display","inline-block")
        .style("width", "110px")
        .style("height", "25px")
        .style("font-size", "16px")
        .html(MathCache("star"));


      // Add pluses and equal signs
      if (i < 5) {
        mathdiv.append("span").style("text-align","center")
        .style("display","inline-block")
        .style("width", "25px")
        .style("height", "25px")
        .style("opacity", 0)
        .html(".")
      } else{
      if (i == 5) {
        mathdiv.append("span").style("text-align","center")
        .style("display","inline-block")
        .style("width", "26px")
        .style("height", "25px")
        .style("opacity", 0)
        .html(".")
        }
      }

    }
  }



  // Render Equations
  for (var i =0; i < 7; i++) {

    var html = MathCache("model");
    if (i != 6) {
      html = renderEquation(hat, wi[i].toPrecision(3), i+1);
    }

    if (i != 6) {
    var equation = mathdiv
      .append("span")
      .attr("class", "draggable-number")
      .style("text-align","center")
      .style("display","inline-block")
      .style("width", "110px")
      .style("height", "50px")
      .style("font-size", "16px")
      .style("cursor", "ew-resize")
      .html(html)
      .call(
      d3.drag()
        .on("start", function() { startpoint = d3.mouse(this)[0] } )
        .on("drag", (function(i) { return function() {
          var xi = xrange(d3.mouse(this)[0] - startpoint) + wi[i]
          witemp = wi.slice(0); witemp[i] = xi
          var str = renderEquation(hat, (witemp[i]).toPrecision(3), i+1);
          d3.select(this).html(str)

          var w = zeros(6); w[i] = xi
          updates[i](numeric.dot(numeric.transpose(U),w))
          updatesum(numeric.dot(numeric.transpose(U),witemp))
        }} )(i) )
        .on("end", function() { wi = witemp })
           )
    } else {
      var equation = mathdiv
        .append("span")
        .style("text-align","center")
        .style("display","inline-block")
        .style("width", "110px")
        .style("height", "50px")
        .style("font-size", "16px")
        .html(html)
    }

    equations.push(equation)

    // Add pluses and equal signs
    if (i < 5) {
      var html = MathCache("plus");
      mathdiv.append("span").style("text-align","center")
      .style("display","inline-block")
      .style("width", "25px")
      .style("height", "50px")
      .style("font-size", "16px")
      .html(html)
    } else{
    if (i == 5) {
      var html = MathCache("equals");
         mathdiv.append("span").style("text-align","center")
      .style("display","inline-block")
      .style("width", "26px")
      .style("height", "50px")
      .style("font-size", "16px")
      .html(html)
      }
    }

  }

  var div = eigensum
    .style("display", "block")
    .style("margin-left","auto")
    .style("margin-right","auto")
    .style("width", 940+"px")
    .style("position", "relative")
    .append("div")

  // Render Polynomials
  var updates = []
  for (var i = 0; i < 6; i++ ){
    var w = zeros(6)
    w[i] = 2
    var update = renderEigenSum(div.append("svg"), x, undefined, function() {}, ["hsl(24, 100%, 50%)", "hsl(24, 100%, 50%)"])
    updates.push(update.poly)
    if (i != 5) {
      div.append("span").html("+").style("position", "relative").style("top", "-54px").style("display","inline-block")
      .style("width", "7px").style("left", "-5px")
    }
  }

  div.append("span").html("=").style("position", "relative").style("top", "-54px").style("position", "relative").style("width", "7px").style("left", "-1px")

  var updatesum = renderEigenSum(div.append("svg"),x, b,
    refit,
    ["black", "black"]).poly

  function updateweights(win) {
    wi = win
    for (var i = 0; i < 6; i++ ){
      var html = renderEquation(hat, win[i].toPrecision(3), i + 1);
      equations[i].html(html)
      var w = zeros(6); w[i] = win[i]
      updates[i](numeric.dot(w,U))
    }
    updatesum(numeric.dot(win,U))

  }

  updateweights(wi)

  return {updates:updates, updatesum:updatesum, updateweights:updateweights}
}

/*

Render the Polynomial fitting widget

*/
function renderEigenSum(svg, xv, b, dragCallback, colors) {

  /*
    Data on eigenvectors

    xv - x values (\xi in paper)
    U - eigenvectors
    Lambda - eigenvalues
  */

  // Linear regression - minimize ||Ax - b||^2. Notation is different in article.
  var A = vandermonde(xv, 25)
  var w = zeros(xv.length)
  /**************************************************************************
    START VISUALIZATION
  ***************************************************************************/

  var width = 110
  var height = 110

  var x = d3.scaleLinear().domain([-1.5,1.5]).range([0, width]);
  var y = d3.scaleLinear().domain([-3,3]).range([height, 0]);

  var valueline = d3.line()
      .x(function(d) { return x(d[0]); })
      .y(function(d) { return y(d[1]); });

  var display_poly = function (weights) {

    w = weights

    eigenpath
      .attr("d", valueline(evalPoly(w) ))
      .style("opacity",1)

    var pd = polyC.selectAll("circle").data(xv).merge(polyC)

    pd.attr("cy", function (d) {
      return y(poly(w, d))
    })

    if (!(datalines === undefined)) {
    datalines
      .attr("x1", function(d,i) { return x(d[0]) })
      .attr("y1", function(d,i) { return y(d[1]) })
      .attr("x2", function(d,i) { return x(d[0]) })
      .attr("y2", function(d,i) { return y(poly(w, d[0])) })
    }
  }

  /*
   * Add eigenvalue plot at the bottom.
   * Some copy and pasted code here, but its only a 1-time thing.
   */

  svg.attr("width", width)
      .attr("height", height)
      .style("padding", "6px")


  var eigensvg = svg.append("svg")
        .attr("width", width)
        .attr("height", height)

  var eigenpath = eigensvg.append("path")
                  .style("stroke-width", "2px")
                  .style("stroke", colors[1])
                  .style("fill","none")

  var polyC = svg.append("g").selectAll("circle").data(xv)
    .enter()
    .append("circle")
    .attr("cx", function(d,i) { return x(d) })
    .attr("cy", function(d,i) { return y(1) })
    .attr("r", 2)
    .style("stroke-width", "1px")
    .style("stroke", colors[1])
    .style("fill", "white")

  if (!(b === undefined)) {

    var voronoi = d3.voronoi()
      .x(function(d) { return x(d[0]); })
      .y(function(d) { return y(d[1]); })
      .extent([[0, 0], [110, 110]]);

    var data = svg.append("g")

    data.selectAll("circle").data(d3.zip(xv,b))
      .enter()
      .append("circle")
      .attr("cx", function(d,i) { return x(d[0]) })
      .attr("cy", function(d,i) { return y(d[1]) })
      .attr("r", 2)
      .style("stroke-width", "8px")
      .style("stroke", "rgba(255,255,255,0.1)")
      .style("fill", colors[0])
      .call(d3.drag()
            .on("drag", function(d,i) {
              var ypos = d3.event.y
              var yval = y.invert(ypos)
              this.setAttribute("cy", ypos)
              b[i] = yval
              dragCallback(b)
              datalines.data(d3.zip(xv,b)).merge(datalines)
                .attr("x1", function(d,i) { return x(d[0]) })
                .attr("y1", function(d,i) { return y(d[1]) })
                .attr("x2", function(d,i) { return x(d[0]) })
                .attr("y2", function(d,i) { return y(poly(w, d[0])) })
            })
          )

    var vongroup = svg.append("g")

    var dragging = false
    vongroup.selectAll("path")
      .data(voronoi.polygons(d3.zip(xv,b)))
      .enter().append("path")
      .attr("d", function(d, i) { return "M" + d.join("L") + "Z"; })
      .datum(function(d, i) { return d.point; })
      .style("stroke", "#2074A0") //I use this to look at how the cells are dispersed as a check
      .style("fill", "white")
      .style("opacity", 0.001)
      // .style("pointer-events", "all")
      .on("mouseover", function(d,i) {
        if (!dragging){
        d3.select(datalinessvg.selectAll("line").nodes()[i]).style("stroke-width", "1px");
        d3.select(data.selectAll("circle").nodes()[i]).style("fill", "red");
        }
      })
      .on("mouseout", function(d,i) {
        if (!dragging) {
        d3.select(datalinessvg.selectAll("line").nodes()[i]).style("stroke-width", "0px");
        d3.select(data.selectAll("circle").nodes()[i]).style("fill", "black");
        }
      })
      // .on("mouseout",  removeTooltip);
      .call(d3.drag()
            .on("drag", function(d,i) {
              dragging = true
              d3.select(datalinessvg.selectAll("line").nodes()[i]).style("stroke-width", "2px");
              d3.select(data.selectAll("circle").nodes()[i]).style("fill", "pink");

              var ypos = d3.event.y
              var yval = y.invert(ypos)
              if (ypos < 0 || ypos > 110) {
                return
              }
              var thisvar = data.selectAll("circle").nodes()[i]
              thisvar.setAttribute("cy", ypos)
              b[i] = yval
              dragCallback(b)
              datalines.data(d3.zip(xv,b)).merge(datalines)
                .attr("x1", function(d,i) { return x(d[0]) })
                .attr("y1", function(d,i) { return y(d[1]) })
                .attr("x2", function(d,i) { return x(d[0]) })
                .attr("y2", function(d,i) { return y(poly(w, d[0])) })
              vongroup.selectAll("path")
                .data(voronoi.polygons(d3.zip(xv,b)))
                .merge(vongroup)
                .attr("d", function(d, i) { return "M" + d.join("L") + "Z"; })
            })
            .on("end", function(d,i) {
              dragging = false
              d3.select(datalinessvg.selectAll("line").nodes()[i]).style("stroke-width", "0px");
              d3.select(data.selectAll("circle").nodes()[i]).style("fill", "black");
            })
          )

    var datalinessvg = svg.append("g")

    var datalines = datalinessvg.selectAll("line").data(d3.zip(xv,b))
      .enter()
      .append("line")
      .attr("x1", function(d,i) { return x(d[0]) })
      .attr("y1", function(d,i) { return y(d[1])+2 })
      .attr("x2", function(d,i) { return x(d[0]) })
      .attr("y2", function(d,i) { return y(0) })
      .style("stroke-width", "0px")
      .style("stroke", "red")

  }

  eigensvg.append("g")
    .attr("class", "grid")
    .attr("transform", "translate(0," + (height/2) + ")")
    .call(d3.axisBottom(x)
        .ticks(1)
        .tickSize(2))

  // Start at some nice looking defaults.
  display_poly(w)

  return {poly:display_poly};
}

