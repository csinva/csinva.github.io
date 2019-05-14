function renderFlowWidget(divin, FlowSigma, M, FlowU) {

  var jetc = d3.scaleLinear().domain([-100,1.5,2,3,4,5,10,60,200,500]).range(colorbrewer.RdYlBu[10]);

  var colorBar = divin.append("div").style("position", "absolute").style("left","490px").style("top", "5px").style("height","45px")

  colorMap( colorBar,
           180,
           jetc,
           d3.scaleLinear().domain([0,100]).range([0, 180]) )

  divin.append("figcaption")
       .style("position", "absolute")
       .style("left", "500px")
       .style("width", "300px")
       .style("top", "70px")
       .html("Each square represents a node, colored by its weight. Edges connect each square to the four neighboring squares.")

  divin.append("figcaption")
       .style("position", "absolute")
       .style("left", "500px")
       .style("width", "100px")
       .style("top", "0px")
       .attr("class", "figtext")
       .html("Weights")

  // sliderGen([320, 130])
  //     .ticks([0,2/FlowSigma[1119]])
  //     .cRadius(5)
  //     .startxval(4)
  //     .shifty(10)
  //     .change(changeStep)(divin.append("div").style("position","relative").style("left", "100px"))

  var slider = divin.append("div").style("position","relative").style("left", "112px")
  /*
  Generate coordinates for 2D Laplacian based on input matrix
  V[i] = (x,y) coordinate of node in 2D space.
  */
  var V = []
  for (var i = 0; i < 16; i++) { for (var j = 0; j < 112; j++) {
    if (M[i][j] == 0){ V.push([j,i]) }
  }}

  /* Generate b in optimization problem x'Ax - b'x */
  var b = zeros(FlowU[0].length); b[448] = 10;  b[560] = 10
  var step = 1.9/(FlowSigma[1119])
  //var iter = geniter(FlowU, FlowSigma, b, step)
  var Ub = numeric.dot(FlowU, b)

  // We can also run this on momentum.
  var iterf = geniterMomentum(FlowU, FlowSigma, b, step, 0.999).iter
  var iter = function(k) { return iterf(k)[1] }

  /**************************************************************************
    START VISUALIZATION
  ***************************************************************************/
  divin.style("position", "relative")

  /* Render the 2D grid of pixels, given a div as input
   * (appends an svg inside the div)
   * Returns a function which allows you to update it */
  function renderGrid(s, transition) {

    var sampleSVG = s.style("display", "block")
      .style("margin-left","auto")
      .style("margin-right","auto")
      .style("width", "920px")
        .append("svg")
        .attr("width", 920)
        .attr("height", 150)

    /* Render discretization of 2D Laplacian*/
    sampleSVG.selectAll("rect")
        .data(V)
        .enter().append("rect")
        .style("fill", function(d,i) { return "white" })
        .attr("height", 7.7)
        .attr("width", 7.7)
        .attr("x", function(d, i){return d[0]*8+ 10})
        .attr("y", function(d, i){return d[1]*8 + 10})

    /* display a function on the laplacian using colormap map*/
    var display = function (x, map) {
      if (transition === undefined){
        sampleSVG.selectAll("rect").style("fill", function(d,i) { return map(x[i]) })
      } else {
        sampleSVG.transition()
          .duration(transition)
          .selectAll("rect")
          .style("fill", function(d,i) { return map(x[i]) })
      }
    }

    return display

  }

  var display = renderGrid(divin)
  /* Render Control Slider */

  // Callbacks
  var showEigen = function (d,i) {
    display(FlowU[i], divergent)
  }

  var cacheval  = -1
  var cacheiter = null
  var onDragSlider = function (i) {
    var i = Math.floor(Math.exp(i-1))
    if (cacheval != i) {
      cacheiter = iter(i)
      cacheval = i
    }
    display(cacheiter, jetc)
  }

  display(iter(100), jetc) // Set it up to a nice looking default

  var barLengths = getStepsConvergence(FlowSigma,step)
    .map( function(i) {return Math.abs(Math.log(i+1)) } ).filter( function(d,i) { return (i < 50) || i%20 == 0 } )

  var slideControl = sliderBarGen(barLengths, function(x) { return Math.exp(x-1)} )
                    .height(281)
                    .linewidth(1.3)
                    .maxX(13.3)
                    .mouseover( function(d,i) { display(FlowU[i], divergent) })
                    .labelFunc(function (d,i) {
                      if (i < 50) {
                        return ((i == 0) ? "Eigenvalue 1" : "") + (( (i+1) % 25 == 0 ) ? (i + 1) : "")
                      } else {
                        return (( (i+1) % 25 == 0 ) ? 20*(i + 1) : "")
                      }
                    })
                    .update(onDragSlider)(divin)

  slideControl.slidera.init()

  function changeStep(alpha, beta) {
    var iteration = geniterMomentum(FlowU, FlowSigma, b, Math.max(alpha/FlowSigma[1119],0.00000001), Math.min(beta,0.999999))
    iterf = iteration.iter
    var barLengths = getStepsConvergence(iteration.maxLambda.map(function(i) { return 1- i}), 1)
      .map( function(i) {return Math.log(Math.max(i,1))} )
      .filter( function(d,i) { return (i < 50) || i%20 == 0 } )
    slideControl.update(barLengths)
    iter = function(k) { return iterf(k)[1] }
    cacheval = -1
    slideControl.slidera.init()
  }
  var reset = slider2D(slider, changeStep, FlowSigma[0], FlowSigma[1119], [1.9,0.00001])

  slider.append("div")
    .attr("class","figtext")
    .style("left", "155px")
    .style("top", "0px")
    .style("position","absolute")
    .style("pointer-events", "none")
    .html("Step-size α =")

  slider.append("div")
    .attr("class","figtext")
    .style("left", "-87px")
    .style("top", "20px")
    .style("position","absolute")
    .style("pointer-events", "none")
    .html("Momentum β = ")

    return reset
}

// <div class="figtext" style="position:absolute; pointer-events:none; top:14px; width:300px; left:729px; height:100px">Step-size α = </div>
// <div class="figtext" style="position:absolute; pointer-events:none; top:35px; width:488px; left:488px; height:100px">Momentum β = </div>
