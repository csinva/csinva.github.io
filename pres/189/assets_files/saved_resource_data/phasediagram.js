function phaseDiagram(divin) {

  var totalIters = 100
  var default_underdamp = 0.97
  var default_overdamp = 0.05

  function getTrace(alpha, beta, coord, sign) {
    var m = []
    var lambda = [1,100]
    var iter = geniterMomentum([[1, 0],[0, 1]], lambda, [1,sign*100], alpha, beta).iter
    // run for 500 iterations
    for (var i = 0; i <= totalIters; i++) {
      var x = numeric.add(iter(i)[1],-sign*1)
      var y = numeric.mul(iter(i)[0],(1/200))
      m.push([y[coord], x[coord]])
    }
    mG = m
    return m
  }

  var textCaptions = ["<center><b style=\"color:black\">Overdamping</b></center><div style=\"margin:5px\"></div> When " + MathCache("beta") + " is too small (e.g. in Gradient Descent, " + MathCache("beta-equals-zero") + "), we're over-damping. The particle is immersed in a viscous fluid which saps it of its kinetic energy at every timestep.",
                      "<b style=\"color:black\"><center>Critical Damping</center></b> <div style=\"margin:5px\"></div>The best value of " + MathCache("beta") + " lies in the middle of the two extremes. This sweet spot happens when the eigenvalues of " + MathCache("r") + " are repeated, when " + MathCache("beta-equals-one-minus") + ".",
                      "<b style=\"color:black\"><center>Underdamping</center></b><div style=\"margin:5px\"></div> When " + MathCache("beta") + " is too large we're under-damping. Here the resistance is too small, and spring oscillates up and down forever, missing the optimal value over and over. "]

  var al = 0.0001
  var optbeta = Math.pow(1 - Math.sqrt(al*100),2)

  var w = 170
  var h = 170
  var a = 1.0

  var axis = [[-a*5,a*5],[-a,a]]
  var width_bar = 620
  var X = d3.scaleLinear().domain([1,0]).range([0, width_bar])

  var valueline = d3.line()
    .x(function(d) { return d[0]; })
    .y(function(d) { return d[1]; });

  var overlay = divin.append("svg")
       .style("position", "absolute")
       .attr("width", 648)
       .attr("height", 520)
       .style("z-index", 10)
       .style("pointer-events", "none")
  renderDraggable(overlay, [320.5, 361+35], [346.5, 378+35], 6, "reaches<tspan x=\"0\" dy=\"1.2em\">optimum</tspan>")
  //renderDraggable(overlay, [102, 312], [107, 310], 6, "initial point:<tspan x=\"0\" dy=\"1.2em\">x = 1, y = 0</tspan>")
  //renderDraggable(overlay, [581.5, 360+35], [597.5, 321+35], 6, "misses optimum")

  // Draw the three phases
  var updateCallbacks = []
  var divs = []
  var ringPath = ringPathGen(5, 0, 0)
  var paths = []
  for (var i = 0; i < 3; i ++ ) {

    var div = divin.append("div")
      .style("position","absolute")
      .style("width",w + "px")
      .style("height",h + "px")
      .style("left", [455, 235, 15][i] + "px")
      .style("top", [110, 110, 110][i] + "px")
      .style("border-top", "solid 1px gray")

    divs.push(div)
    var z = div.node()

    var divx = z.offsetLeft + z.offsetWidth/2
    var divy = z.offsetTop
    var path = overlay.append("path")
                  .style("stroke", "grey")
                  .style("stroke-width", "1px")
                  .style("fill", "none")
                  .attr("stroke-dasharray", "5,3")
                  .attr("opacity", 0.7)

    if (i == 0) {

      var updateAnnotationOverDamp = (function(pathin, divxin, divyin) {
        return function(x,y,d) {
          pathin.transition().duration(d).attr("d", ringPath([x+10,y],[divxin,divyin]).d)
        }
      })(path, divx, divy)
      updateAnnotationOverDamp(X(default_overdamp), 30, 0)
    }

    if (i == 1) {
      path.attr("d", ringPath([X(0.8)+5,30],[divx - 40,divy -40]).d + "L" + divx + "," + divy + " ")
    }

    if (i == 2) {

      var updateAnnotationUnderDamp = (function(pathin, divxin, divyin) {
        return function(x,y,d) {
          pathin.transition().duration(d).attr("d", ringPath([x+10,y],[divxin,divyin]).d)
        }
      })(path, divx, divy)
      updateAnnotationUnderDamp(X(default_underdamp), 30, 0)

    }

    paths.push(path)

    div.append("figcaption")
      .style("position","absolute")
      .style("width", "180px")
      .style("height", "200px")
      .style("text-align", "left")
      .style("top", [10, 10, 10][i] + "px")
      .style("left", "00px")
      .html(textCaptions[i])

    var svg = div.append("svg")
                .style("position", 'absolute')
                .style("left", 0)
                .style("top", "200px")
                .style("width", w)
                .style("height", h)
      .style("border-radius", "5px")

    svg.append("g").attr("class", "grid")
      .attr("transform", "translate(0," + h/2 +")")
      .attr("opacity", 0.2)
      .call(d3.axisBottom(X).ticks(0).tickSize(0))

    svg.append("g").attr("class", "grid")
      .attr("transform", "translate(" + w/2 + ",0)")
      .attr("opacity", 0.2)
      .call(d3.axisLeft(X).ticks(0).tickSize(0))

    var colorRange = d3.scaleLinear().domain([0, totalIters/16, totalIters/2]).range(colorbrewer.OrRd[3])

    var Xaxis = d3.scaleLinear().domain(axis[0]).range([0, w])
    var Yaxis = d3.scaleLinear().domain(axis[1]).range([0, h])

    var update = plot2dGen(Xaxis, Yaxis, colorRange)
                  .pathOpacity(1)
                  .pathWidth(1.5)
                  .circleRadius(1.5)
                  .stroke(colorbrewer.OrRd[3][0])(svg)

    update(getTrace(al, [0.01, optbeta + 0.0001 , default_underdamp][i], 1,1))
    updateCallbacks.push(update)

  }

  var linesvg = divin
    .append("svg")

  // Axis
  var axis = linesvg.append("g")
    .attr("class", "figtext")
    .attr("opacity", 0.3)
    .attr("transform", "translate(0,32)")
    .call(d3.axisBottom(X)
      .ticks(5)
      .tickSize(5))

  axis.selectAll("path").remove()
  axis.select("text").style("text-anchor", "start");

  var html = MathCache("beta");
  // Axis
  linesvg.append("text")
    .attr("class", "figtext")
    .attr("opacity", 1)
    .attr("transform", "translate(0,12)")
    .html("Momentum  β")

  linesvg.style("position","absolute")
    .style("width", "920px")
    .style("height", "570px")
    .style("left", "10px")
    .append("line")
    .attr("x1", 0)
    .attr("y1", 30)
    .attr("x2", 0 + width_bar)
    .attr("y2", 30)
    .style("border", "solid 2px black")
    .style("stroke", "#CCC")
    .style("fill", "white")
    .style("stroke-width", "1.5px")

  var underdamp = linesvg
        .append("circle")
        .attr("cx", X(default_overdamp))
        .attr("cy", 30)
        .attr("r", 6)
        .style("fill", "#ff6600")

  var criticaldamp = linesvg
        .append("circle")
        .attr("cx", X(optbeta))
        .attr("cy", 30)
        .attr("r", 6)
        .style("fill", "#ff6600")

  var overdamp = linesvg
        .append("circle")
        .attr("cx", X(default_underdamp))
        .attr("cy", 30)
        .attr("r", 6)
        .style("fill", "#ff6600")

  var prevState = ""

  linesvg.style("position","absolute")
    .style("width", "920px")
    .style("height", "570px")
    .append("line")
    .attr("x1", 0)
    .attr("y1", 30)
    .attr("x2", 0 + width_bar)
    .attr("y2", 30)
    .style("border", "solid 2px black")
    .style("stroke", "black")
    .style("fill", "white")
    .style("opacity", 0)
    .style("z-index", 3)
    .style("stroke-width", "40px")
    .on("mousemove", function () {

      var pt = d3.mouse(this)
      var beta = X.invert(pt[0])
      if (beta < optbeta) {
        underdamp.attr("cx", pt[0])
        updateCallbacks[0](getTrace(al, X.invert(pt[0]), 1,1))

        overdamp.attr("cx", X(default_underdamp))
        updateAnnotationOverDamp(pt[0], 30, 0)

        if (prevState != "Over"){
          divs[0].style("opacity",1)
          divs[1].style("opacity",0.2)
          divs[2].style("opacity",0.2)

          updateCallbacks[2](getTrace(al, default_underdamp, 1,1))
          updateAnnotationUnderDamp(X(default_underdamp), 30, 20)
        }
        prevState = "Over"
      }
      if (beta > optbeta) {
        overdamp.attr("cx", pt[0])
        underdamp.attr("cx", X(default_overdamp))
        updateCallbacks[2](getTrace(al, Math.min(X.invert(pt[0]),1), 1,1))

        updateAnnotationUnderDamp(pt[0], 30, 0)

        if (prevState != "Under") {
          divs[0].style("opacity",0.2)
          divs[1].style("opacity",0.2)
          divs[2].style("opacity",1)

          updateCallbacks[0](getTrace(al, default_overdamp, 1,1))
          updateAnnotationOverDamp(X(default_overdamp), 30, 20)
        }
        prevState = "Under"
      }

    })
    .on("mouseout", function () {
        divs[0].style("opacity",1)
        divs[1].style("opacity",1)
        divs[2].style("opacity",1)

      underdamp.transition().duration(50).attr("cx", X(default_overdamp))
      updateCallbacks[0](getTrace(al, default_overdamp, 1,1))

      overdamp.transition().duration(50).attr("cx", X(default_underdamp))
      updateCallbacks[2](getTrace(al, default_underdamp, 1,1))

      updateAnnotationUnderDamp(X(default_underdamp), 30, 50)
      updateAnnotationOverDamp(X(default_overdamp), 30, 50)
      prevState = ""

    })



}
