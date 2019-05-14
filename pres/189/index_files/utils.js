/****************************************************************************
  COLORMAPS AND STUFF
****************************************************************************/

var colorbrewer={YlGn:{3:["#f7fcb9","#addd8e","#31a354"],4:["#ffffcc","#c2e699","#78c679","#238443"],5:["#ffffcc","#c2e699","#78c679","#31a354","#006837"],6:["#ffffcc","#d9f0a3","#addd8e","#78c679","#31a354","#006837"],7:["#ffffcc","#d9f0a3","#addd8e","#78c679","#41ab5d","#238443","#005a32"],8:["#ffffe5","#f7fcb9","#d9f0a3","#addd8e","#78c679","#41ab5d","#238443","#005a32"],9:["#ffffe5","#f7fcb9","#d9f0a3","#addd8e","#78c679","#41ab5d","#238443","#006837","#004529"]},YlGnBu:{3:["#edf8b1","#7fcdbb","#2c7fb8"],4:["#ffffcc","#a1dab4","#41b6c4","#225ea8"],5:["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"],6:["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#2c7fb8","#253494"],7:["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#0c2c84"],8:["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#0c2c84"],9:["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"]},GnBu:{3:["#e0f3db","#a8ddb5","#43a2ca"],4:["#f0f9e8","#bae4bc","#7bccc4","#2b8cbe"],5:["#f0f9e8","#bae4bc","#7bccc4","#43a2ca","#0868ac"],6:["#f0f9e8","#ccebc5","#a8ddb5","#7bccc4","#43a2ca","#0868ac"],7:["#f0f9e8","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#08589e"],8:["#f7fcf0","#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#08589e"],9:["#f7fcf0","#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#0868ac","#084081"]},BuGn:{3:["#e5f5f9","#99d8c9","#2ca25f"],4:["#edf8fb","#b2e2e2","#66c2a4","#238b45"],5:["#edf8fb","#b2e2e2","#66c2a4","#2ca25f","#006d2c"],6:["#edf8fb","#ccece6","#99d8c9","#66c2a4","#2ca25f","#006d2c"],7:["#edf8fb","#ccece6","#99d8c9","#66c2a4","#41ae76","#238b45","#005824"],8:["#f7fcfd","#e5f5f9","#ccece6","#99d8c9","#66c2a4","#41ae76","#238b45","#005824"],9:["#f7fcfd","#e5f5f9","#ccece6","#99d8c9","#66c2a4","#41ae76","#238b45","#006d2c","#00441b"]},PuBuGn:{3:["#ece2f0","#a6bddb","#1c9099"],4:["#f6eff7","#bdc9e1","#67a9cf","#02818a"],5:["#f6eff7","#bdc9e1","#67a9cf","#1c9099","#016c59"],6:["#f6eff7","#d0d1e6","#a6bddb","#67a9cf","#1c9099","#016c59"],7:["#f6eff7","#d0d1e6","#a6bddb","#67a9cf","#3690c0","#02818a","#016450"],8:["#fff7fb","#ece2f0","#d0d1e6","#a6bddb","#67a9cf","#3690c0","#02818a","#016450"],9:["#fff7fb","#ece2f0","#d0d1e6","#a6bddb","#67a9cf","#3690c0","#02818a","#016c59","#014636"]},PuBu:{3:["#ece7f2","#a6bddb","#2b8cbe"],4:["#f1eef6","#bdc9e1","#74a9cf","#0570b0"],5:["#f1eef6","#bdc9e1","#74a9cf","#2b8cbe","#045a8d"],6:["#f1eef6","#d0d1e6","#a6bddb","#74a9cf","#2b8cbe","#045a8d"],7:["#f1eef6","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#034e7b"],8:["#fff7fb","#ece7f2","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#034e7b"],9:["#fff7fb","#ece7f2","#d0d1e6","#a6bddb","#74a9cf","#3690c0","#0570b0","#045a8d","#023858"]},BuPu:{3:["#e0ecf4","#9ebcda","#8856a7"],4:["#edf8fb","#b3cde3","#8c96c6","#88419d"],5:["#edf8fb","#b3cde3","#8c96c6","#8856a7","#810f7c"],6:["#edf8fb","#bfd3e6","#9ebcda","#8c96c6","#8856a7","#810f7c"],7:["#edf8fb","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#6e016b"],8:["#f7fcfd","#e0ecf4","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#6e016b"],9:["#f7fcfd","#e0ecf4","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#810f7c","#4d004b"]},RdPu:{3:["#fde0dd","#fa9fb5","#c51b8a"],4:["#feebe2","#fbb4b9","#f768a1","#ae017e"],5:["#feebe2","#fbb4b9","#f768a1","#c51b8a","#7a0177"],6:["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#c51b8a","#7a0177"],7:["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177"],8:["#fff7f3","#fde0dd","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177"],9:["#fff7f3","#fde0dd","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177","#49006a"]},PuRd:{3:["#e7e1ef","#c994c7","#dd1c77"],4:["#f1eef6","#d7b5d8","#df65b0","#ce1256"],5:["#f1eef6","#d7b5d8","#df65b0","#dd1c77","#980043"],6:["#f1eef6","#d4b9da","#c994c7","#df65b0","#dd1c77","#980043"],7:["#f1eef6","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#91003f"],8:["#f7f4f9","#e7e1ef","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#91003f"],9:["#f7f4f9","#e7e1ef","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#980043","#67001f"]},OrRd:{3:["#fee8c8","#fdbb84","#e34a33"],4:["#fef0d9","#fdcc8a","#fc8d59","#d7301f"],5:["#fef0d9","#fdcc8a","#fc8d59","#e34a33","#b30000"],6:["#fef0d9","#fdd49e","#fdbb84","#fc8d59","#e34a33","#b30000"],7:["#fef0d9","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#990000"],8:["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#990000"],9:["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#b30000","#7f0000"]},YlOrRd:{3:["#ffeda0","#feb24c","#f03b20"],4:["#ffffb2","#fecc5c","#fd8d3c","#e31a1c"],5:["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026"],6:["#ffffb2","#fed976","#feb24c","#fd8d3c","#f03b20","#bd0026"],7:["#ffffb2","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#b10026"],8:["#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#b10026"],9:["#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"]},YlOrBr:{3:["#fff7bc","#fec44f","#d95f0e"],4:["#ffffd4","#fed98e","#fe9929","#cc4c02"],5:["#ffffd4","#fed98e","#fe9929","#d95f0e","#993404"],6:["#ffffd4","#fee391","#fec44f","#fe9929","#d95f0e","#993404"],7:["#ffffd4","#fee391","#fec44f","#fe9929","#ec7014","#cc4c02","#8c2d04"],8:["#ffffe5","#fff7bc","#fee391","#fec44f","#fe9929","#ec7014","#cc4c02","#8c2d04"],9:["#ffffe5","#fff7bc","#fee391","#fec44f","#fe9929","#ec7014","#cc4c02","#993404","#662506"]},Purples:{3:["#efedf5","#bcbddc","#756bb1"],4:["#f2f0f7","#cbc9e2","#9e9ac8","#6a51a3"],5:["#f2f0f7","#cbc9e2","#9e9ac8","#756bb1","#54278f"],6:["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#756bb1","#54278f"],7:["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"],8:["#fcfbfd","#efedf5","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"],9:["#fcfbfd","#efedf5","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#54278f","#3f007d"]},Blues:{3:["#deebf7","#9ecae1","#3182bd"],4:["#eff3ff","#bdd7e7","#6baed6","#2171b5"],5:["#eff3ff","#bdd7e7","#6baed6","#3182bd","#08519c"],6:["#eff3ff","#c6dbef","#9ecae1","#6baed6","#3182bd","#08519c"],7:["#eff3ff","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"],8:["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"],9:["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#08519c","#08306b"]},Greens:{3:["#e5f5e0","#a1d99b","#31a354"],4:["#edf8e9","#bae4b3","#74c476","#238b45"],5:["#edf8e9","#bae4b3","#74c476","#31a354","#006d2c"],6:["#edf8e9","#c7e9c0","#a1d99b","#74c476","#31a354","#006d2c"],7:["#edf8e9","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#005a32"],8:["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#005a32"],9:["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#006d2c","#00441b"]},Oranges:{3:["#fee6ce","#fdae6b","#e6550d"],4:["#feedde","#fdbe85","#fd8d3c","#d94701"],5:["#feedde","#fdbe85","#fd8d3c","#e6550d","#a63603"],6:["#feedde","#fdd0a2","#fdae6b","#fd8d3c","#e6550d","#a63603"],7:["#feedde","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#8c2d04"],8:["#fff5eb","#fee6ce","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#8c2d04"],9:["#fff5eb","#fee6ce","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#a63603","#7f2704"]},Reds:{3:["#fee0d2","#fc9272","#de2d26"],4:["#fee5d9","#fcae91","#fb6a4a","#cb181d"],5:["#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"],6:["#fee5d9","#fcbba1","#fc9272","#fb6a4a","#de2d26","#a50f15"],7:["#fee5d9","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#99000d"],8:["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#99000d"],9:["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#a50f15","#67000d"]},Greys:{3:["#f0f0f0","#bdbdbd","#636363"],4:["#f7f7f7","#cccccc","#969696","#525252"],5:["#f7f7f7","#cccccc","#969696","#636363","#252525"],6:["#f7f7f7","#d9d9d9","#bdbdbd","#969696","#636363","#252525"],7:["#f7f7f7","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525"],8:["#ffffff","#f0f0f0","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525"],9:["#ffffff","#f0f0f0","#d9d9d9","#bdbdbd","#969696","#737373","#525252","#252525","#000000"]},PuOr:{3:["#f1a340","#f7f7f7","#998ec3"],4:["#e66101","#fdb863","#b2abd2","#5e3c99"],5:["#e66101","#fdb863","#f7f7f7","#b2abd2","#5e3c99"],6:["#b35806","#f1a340","#fee0b6","#d8daeb","#998ec3","#542788"],7:["#b35806","#f1a340","#fee0b6","#f7f7f7","#d8daeb","#998ec3","#542788"],8:["#b35806","#e08214","#fdb863","#fee0b6","#d8daeb","#b2abd2","#8073ac","#542788"],9:["#b35806","#e08214","#fdb863","#fee0b6","#f7f7f7","#d8daeb","#b2abd2","#8073ac","#542788"],10:["#7f3b08","#b35806","#e08214","#fdb863","#fee0b6","#d8daeb","#b2abd2","#8073ac","#542788","#2d004b"],11:["#7f3b08","#b35806","#e08214","#fdb863","#fee0b6","#f7f7f7","#d8daeb","#b2abd2","#8073ac","#542788","#2d004b"]},BrBG:{3:["#d8b365","#f5f5f5","#5ab4ac"],4:["#a6611a","#dfc27d","#80cdc1","#018571"],5:["#a6611a","#dfc27d","#f5f5f5","#80cdc1","#018571"],6:["#8c510a","#d8b365","#f6e8c3","#c7eae5","#5ab4ac","#01665e"],7:["#8c510a","#d8b365","#f6e8c3","#f5f5f5","#c7eae5","#5ab4ac","#01665e"],8:["#8c510a","#bf812d","#dfc27d","#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e"],9:["#8c510a","#bf812d","#dfc27d","#f6e8c3","#f5f5f5","#c7eae5","#80cdc1","#35978f","#01665e"],10:["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"],11:["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#f5f5f5","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"]},PRGn:{3:["#af8dc3","#f7f7f7","#7fbf7b"],4:["#7b3294","#c2a5cf","#a6dba0","#008837"],5:["#7b3294","#c2a5cf","#f7f7f7","#a6dba0","#008837"],6:["#762a83","#af8dc3","#e7d4e8","#d9f0d3","#7fbf7b","#1b7837"],7:["#762a83","#af8dc3","#e7d4e8","#f7f7f7","#d9f0d3","#7fbf7b","#1b7837"],8:["#762a83","#9970ab","#c2a5cf","#e7d4e8","#d9f0d3","#a6dba0","#5aae61","#1b7837"],9:["#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837"],10:["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b"],11:["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b"]},PiYG:{3:["#e9a3c9","#f7f7f7","#a1d76a"],4:["#d01c8b","#f1b6da","#b8e186","#4dac26"],5:["#d01c8b","#f1b6da","#f7f7f7","#b8e186","#4dac26"],6:["#c51b7d","#e9a3c9","#fde0ef","#e6f5d0","#a1d76a","#4d9221"],7:["#c51b7d","#e9a3c9","#fde0ef","#f7f7f7","#e6f5d0","#a1d76a","#4d9221"],8:["#c51b7d","#de77ae","#f1b6da","#fde0ef","#e6f5d0","#b8e186","#7fbc41","#4d9221"],9:["#c51b7d","#de77ae","#f1b6da","#fde0ef","#f7f7f7","#e6f5d0","#b8e186","#7fbc41","#4d9221"],10:["#8e0152","#c51b7d","#de77ae","#f1b6da","#fde0ef","#e6f5d0","#b8e186","#7fbc41","#4d9221","#276419"],11:["#8e0152","#c51b7d","#de77ae","#f1b6da","#fde0ef","#f7f7f7","#e6f5d0","#b8e186","#7fbc41","#4d9221","#276419"]},RdBu:{3:["#ef8a62","#f7f7f7","#67a9cf"],4:["#ca0020","#f4a582","#92c5de","#0571b0"],5:["#ca0020","#f4a582","#f7f7f7","#92c5de","#0571b0"],6:["#b2182b","#ef8a62","#fddbc7","#d1e5f0","#67a9cf","#2166ac"],7:["#b2182b","#ef8a62","#fddbc7","#f7f7f7","#d1e5f0","#67a9cf","#2166ac"],8:["#b2182b","#d6604d","#f4a582","#fddbc7","#d1e5f0","#92c5de","#4393c3","#2166ac"],9:["#b2182b","#d6604d","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#92c5de","#4393c3","#2166ac"],10:["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#d1e5f0","#92c5de","#4393c3","#2166ac","#053061"],11:["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#92c5de","#4393c3","#2166ac","#053061"]},RdGy:{3:["#ef8a62","#ffffff","#999999"],4:["#ca0020","#f4a582","#bababa","#404040"],5:["#ca0020","#f4a582","#ffffff","#bababa","#404040"],6:["#b2182b","#ef8a62","#fddbc7","#e0e0e0","#999999","#4d4d4d"],7:["#b2182b","#ef8a62","#fddbc7","#ffffff","#e0e0e0","#999999","#4d4d4d"],8:["#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d"],9:["#b2182b","#d6604d","#f4a582","#fddbc7","#ffffff","#e0e0e0","#bababa","#878787","#4d4d4d"],10:["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"],11:["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#ffffff","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"]},RdYlBu:{3:["#fc8d59","#ffffbf","#91bfdb"],4:["#d7191c","#fdae61","#abd9e9","#2c7bb6"],5:["#d7191c","#fdae61","#ffffbf","#abd9e9","#2c7bb6"],6:["#d73027","#fc8d59","#fee090","#e0f3f8","#91bfdb","#4575b4"],7:["#d73027","#fc8d59","#fee090","#ffffbf","#e0f3f8","#91bfdb","#4575b4"],8:["#d73027","#f46d43","#fdae61","#fee090","#e0f3f8","#abd9e9","#74add1","#4575b4"],9:["#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4"],10:["#a50026","#d73027","#f46d43","#fdae61","#fee090","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"],11:["#a50026","#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"]},Spectral:{3:["#fc8d59","#ffffbf","#99d594"],4:["#d7191c","#fdae61","#abdda4","#2b83ba"],5:["#d7191c","#fdae61","#ffffbf","#abdda4","#2b83ba"],6:["#d53e4f","#fc8d59","#fee08b","#e6f598","#99d594","#3288bd"],7:["#d53e4f","#fc8d59","#fee08b","#ffffbf","#e6f598","#99d594","#3288bd"],8:["#d53e4f","#f46d43","#fdae61","#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd"],9:["#d53e4f","#f46d43","#fdae61","#fee08b","#ffffbf","#e6f598","#abdda4","#66c2a5","#3288bd"],10:["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"],11:["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#ffffbf","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"]},RdYlGn:{3:["#fc8d59","#ffffbf","#91cf60"],4:["#d7191c","#fdae61","#a6d96a","#1a9641"],5:["#d7191c","#fdae61","#ffffbf","#a6d96a","#1a9641"],6:["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],7:["#d73027","#fc8d59","#fee08b","#ffffbf","#d9ef8b","#91cf60","#1a9850"],8:["#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850"],9:["#d73027","#f46d43","#fdae61","#fee08b","#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850"],10:["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"],11:["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#ffffbf","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"]},Accent:{3:["#7fc97f","#beaed4","#fdc086"],4:["#7fc97f","#beaed4","#fdc086","#ffff99"],5:["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0"],6:["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f"],7:["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17"],8:["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17","#666666"]},Dark2:{3:["#1b9e77","#d95f02","#7570b3"],4:["#1b9e77","#d95f02","#7570b3","#e7298a"],5:["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e"],6:["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02"],7:["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d"],8:["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"]},Paired:{3:["#a6cee3","#1f78b4","#b2df8a"],4:["#a6cee3","#1f78b4","#b2df8a","#33a02c"],5:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99"],6:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c"],7:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f"],8:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00"],9:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6"],10:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a"],11:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99"],12:["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99","#b15928"]},Pastel1:{3:["#fbb4ae","#b3cde3","#ccebc5"],4:["#fbb4ae","#b3cde3","#ccebc5","#decbe4"],5:["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6"],6:["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc"],7:["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd"],8:["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec"],9:["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec","#f2f2f2"]},Pastel2:{3:["#b3e2cd","#fdcdac","#cbd5e8"],4:["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4"],5:["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9"],6:["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9","#fff2ae"],7:["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9","#fff2ae","#f1e2cc"],8:["#b3e2cd","#fdcdac","#cbd5e8","#f4cae4","#e6f5c9","#fff2ae","#f1e2cc","#cccccc"]},Set1:{3:["#e41a1c","#377eb8","#4daf4a"],4:["#e41a1c","#377eb8","#4daf4a","#984ea3"],5:["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"],6:["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"],7:["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628"],8:["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf"],9:["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf","#999999"]},Set2:{3:["#66c2a5","#fc8d62","#8da0cb"],4:["#66c2a5","#fc8d62","#8da0cb","#e78ac3"],5:["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854"],6:["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f"],7:["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494"],8:["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3"]},Set3:{3:["#8dd3c7","#ffffb3","#bebada"],4:["#8dd3c7","#ffffb3","#bebada","#fb8072"],5:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3"],6:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462"],7:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69"],8:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5"],9:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9"],10:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd"],11:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd","#ccebc5"],12:["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd","#ccebc5","#ffed6f"]}};
var jetc = d3.scaleLinear().domain([-100,1.5,2,3,4,5,10,60,200,500]).range(colorbrewer.RdYlBu[10]);
var divergent = d3.scaleLinear().domain([-0.03,0,0.03]).range(["#d7191c", "#ffffbf","#2b83ba"]);


/****************************************************************************
  "GENERIC" WIDGETS!
****************************************************************************/

/* Custom slider with ticks, tooltips and all that jazz. */
function sliderGen(dims) {

  var onMouseover = function() {}
  var onMouseout  = function() {}
  var onChange    = function() {}
  var ticks = [0, 1]
  var margin = {right: 50, left: 50}
  var curr_xval = 0
  var tooltipcallback = undefined
  var cr = 9
  var tickwidth = 1.5
  var tickheight = 7
  var ticksym = false // |---|-- vs |____|__
  var shifty = -10
  var ticktitles = function(d,i) { return round(d) }
  var showticks = true
  var default_xval = 0

  function renderSlider(divin) {

    var tip = d3.tip()
      .attr('class', 'd3-tip')
      .offset([-12, 0])

    var minLambda = Math.min.apply(null, ticks.filter(function(i) {return !isNaN(i)}))
    var maxLambda = Math.max.apply(null, ticks.filter(function(i) {return !isNaN(i)}))
    var width = dims[0] - margin.left - margin.right
    var height = dims[1]

    var svg = divin.append("svg")
	                  .attr("width", dims[0])
	                  .attr("height",dims[1])
	                  .style("position", "relative")
	                  .append("g")
	                  .attr("transform", "translate(0," + shifty + ")")

    var x = d3.scaleLinear()
        .domain([0, maxLambda])
        .range([0, width])
        .clamp(true);

    var slidersvg = svg.append("g")
        .attr("class", "slidersvg")
        .attr("transform", "translate(" + margin.left + "," + height / 2 + ")");

    slidersvg.call(tip)

    var dragger = slidersvg.append("line")
        .attr("class", "track")
        .attr("x1", x.range()[0])
        .attr("x2", x.range()[1])
      .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
        .attr("class", "track-inset")
      .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
        .attr("class", "track-overlay")
        .call(d3.drag()
          .on("start.interrupt", function() { slidersvg.interrupt(); })
          .on("start drag", function() {
            var xval = x.invert(d3.event.x)
            handle.attr("transform", "translate(" + x(xval) + ",0)" );
            curr_xval = xval
            onChange(xval, handle)
          }));

    var ticksvg = slidersvg.append("g")

    if (showticks) {
	    ticksvg.selectAll("rect")
	      .data(ticks, function(d,i) {return i})
	      .enter().append("rect")
	      .attr("x", function(i) { return isNaN(i) ? -100: x(i) - tickwidth/2})
	      .attr("y", 9)
	      .attr("width", tickwidth )
	      .attr("height", function(d, i) { return (isNaN(i)) ? 0: ticksym ? tickheight*2: tickheight;} )
	      .attr("opacity",0.2 )

	    ticksvg.selectAll("text")
	      .data(ticks, function(d,i) {return i})
	      .enter().append("text")
				  .attr("class", "ticktext")
				  .attr("opacity", 0.3)
				  .attr("text-anchor", "middle")
			      .attr("transform", function(i) { return "translate(" + (isNaN(i) ? -100: x(i) - tickwidth/2 + 1) + "," + (tickwidth*2 + 24) + ")" })
			      .html(ticktitles)
	}
    ticksvg.selectAll("circle")
      .data(ticks,function(d,i) {return i})
      .enter()
      .append("circle", ".track-overlay")
      .attr("cx", function(i) { return isNaN(i) ? -100: x(i);})
      .attr("cy", 0)
      .attr("r", 3)
      .attr("opacity", 0.0)
      .on("mouseover", function(i,k) {
        this.setAttribute("opacity", "0.2");
        if (!(tooltipcallback === undefined)) {
          var tooltip = tooltipcallback(k)
          if (tooltip != false) { tip.show('<span>' + tooltip + '</span>') }
        }
        onMouseover(i,k)
      })
      .on("mouseout", function(i,k) {
        if (!(tooltipcallback === undefined)) { tip.hide() }
        this.setAttribute("opacity", "0");
        onMouseout(i,k)
      })
      .on("click", function(lambda){
        var xval = lambda
        curr_xval = xval
        handle.attr("transform", "translate(" + x(xval) + ",0)" );
        onChange(xval, handle)
      })

    /*
      Update the ticks
    */
    var updateTicks = function(newticks) {

      var d1 = ticksvg.selectAll("rect")
        .data(newticks,function(d,i) {return i})

      d1.exit().remove()
      d1.merge(d1).transition().duration(50)
        .attr("x", function(i) { return isNaN(i) ? -100: x(i) - 0.5})

      var d2 = ticksvg.selectAll("circle")
        .data(newticks,function(d,i) {return i})
      d2.exit().remove()
      d2.merge(d2)
        .attr("cx", function(i) { return isNaN(i) ? -100: x(i);})

    }

    var handle = slidersvg.insert("g", ".track-overlay")
        .attr("transform", "translate(" + x(curr_xval) + ",0)" );

    handle.insert("circle")
        .attr("class", "handle")
        .attr("r", cr)
        .style("fill", "#ff6600")
        .style("fill-opacity", 1)
        .style("stroke", "white")
        .call(d3.drag()
          .on("start.interrupt", function() { slidersvg.interrupt(); })
          .on("start drag", function() {
            var xval = x.invert(d3.mouse(dragger.node())[0])
            handle.attr("transform", "translate(" + x(xval) + ",0)" );
            curr_xval = xval
            onChange(xval, handle)
          }));

    handle.insert("text")
          .attr("transform", "translate(0,22)")
          .attr("text-anchor","middle")
          .style("font-size", "10px")

    handle.moveToFront()
    return {xval : function() { return curr_xval },
    		tick : updateTicks,
    		init:function() {
		        handle.attr("transform", "translate(" + x(default_xval) + ",0)" );
		        onChange(default_xval, handle)
    		}
    }

  }

  renderSlider.ticktitles = function(f) {
    ticktitles = f
    return renderSlider
  }

  renderSlider.mouseover = function(f) {
    onMouseover = f
    return renderSlider
  }

  renderSlider.mouseout = function(f) {
    onMouseOut = f
    return renderSlider
  }

  renderSlider.change = function(f) {
    onChange = f
    return renderSlider
  }

  renderSlider.margin = function(m) {
    margin = m
    return renderSlider
  }

  renderSlider.ticks = function(m) {
    ticks = m
    return renderSlider
  }

  renderSlider.startxval = function(m) {
    curr_xval = m
    default_xval = m
    return renderSlider
  }

  renderSlider.margins = function(l, r) {
    margin = {right: l, left: r}
    return renderSlider
  }

  renderSlider.tooltip = function(f) {
    tooltipcallback = f
    return renderSlider
  }

  renderSlider.cRadius = function(m) {
    cr = m
    return renderSlider
  }

  renderSlider.tickConfig = function(_1,_2,_3) {
    tickwidth = _1
    tickheight = _2
    ticksym = _3 // |---|-- vs |____|__
    return renderSlider
  }

  renderSlider.shifty = function(_) {
  	shifty = _
    return renderSlider
  }

  renderSlider.showticks = function(_) {
    showticks = _
    return renderSlider
  }

  renderSlider.shifty = function(_) {
  	shifty = _; return renderSlider
  }
  return renderSlider
}

/* Generate "stick" graph */
function stemGraphGen(graphWidth, graphHeight, n) {

  var borderTop = 20
  var borderLeft = -5
  var axis = [-1, 1]
  var ylabel = MathCache("x-i-k");
  var ylabelsize = "13px"
  var r1 = 2
  var r2 = 0
  var ticks = 10;

  function renderGraph(outdiv) {

    outdiv.append("span")
      .style("top", (graphHeight/2 + borderTop/2) + "px")
      .style("left", (-graphHeight/2 - 17) + "px" )
      .style("position", "absolute")
      .style("width", graphHeight + "px")
      .style("height", "20px")
      .style("position", "absolute")
      .style("transform", "rotate(-90deg)")
      .style("text-align", "center")
      .style("font-size", ylabelsize)
      .html(ylabel)

    var svg = outdiv.append("svg")
          .attr("width", graphWidth)
          .attr("height", graphHeight)
          // .style("border", "black solid 1px")
          // .style("box-shadow","0px 0px 10px rgba(0, 0, 0, 0.2)")
          .style("position", "absolute")
          .style("top", borderTop)
          .style("left", borderLeft)
          .style("border-radius", "2px")

    var x = d3.scaleLinear().domain([0,n]).range([10, graphWidth-10]);
    var y = d3.scaleLinear().domain(axis).range([graphHeight, 0]);
    var cscale = d3.scaleLinear().domain([axis[0], 0, axis[1]]).range(["black", "black","black"]);
    var valueline = d3.line()
      .x(function(d,i) { return x(i); })
      .y(function(d)   { return y(d); });

    // Initialize the data

    function initData(color, r) {
      var dots = svg.append("g")
      var dotsdata = dots.selectAll("circle").data(zeros(n), function (d, i) { return i })

      dotsdata.enter()
        .append("circle")
        .attr("cx", function(d,i) { return x(i) })
        .attr("cy", function(d) { return y(d) })
        .attr("r", r)
        .style("fill", "darkblue")

      dotsdata.enter()
        .append("line")
        .attr("x1", function(d,i) { return x(i) })
        .attr("x2", function(d,i) { return x(i) })
        .attr("y1", function(d) { return y(d) })
        .attr("y2", y(0))
        .style("stroke",color )
        .attr("opacity", 1)
        .style("stroke-width",1.5)

      return dots;
    }

    var dots1 = initData("#999",r1)
    var dots2 = initData(colorbrewer.RdPu[5][2],r2)

    function updateData(dots, data) {
      dots.selectAll("circle").data(data)
        .attr("cx", function(d,i) { return x(i) })
        .attr("cy", function(d) { return y(Math.min(d,20)) })
      dots.selectAll("line").data(data)
        .attr("y1", function(d) { return y(Math.min(d,20)) })
    }

    var updatePath = function(d1, d2) {
      updateData(dots1, d1)
      if (!(d2 === undefined)) { updateData(dots2, d2) }
    }

    // Add x axis
    svg.append("g")
      .attr("class", "grid")
      .attr("transform", "translate(0," + y(0) + ")")
      .call(d3.axisBottom(x)
        .ticks(ticks)
        .tickSize(2)
        .tickFormat(""))

    return updatePath
  }

  renderGraph.borderTop = function(_) {
  	borderTop = _;
  	return renderGraph;
  }

  renderGraph.axis = function(a) {
    axis = a;
    return renderGraph
  }

  renderGraph.ylabel = function(a) {
    ylabel = a;
    return renderGraph
  }

  renderGraph.radius1 = function(a) {
    r1 = a;
    return renderGraph
  }

  renderGraph.labelSize = function(s) {
  	ylabelsize = s;
  	return renderGraph
  }

  renderGraph.numTicks = function(s) {
  	ticks = s;
  	return renderGraph
  }

  return renderGraph
}

/* Render a stacked graph. D*/
function stackedBarchartGen(n, m) {

  var axis = [0,1.53]
  var translatex = 110
  var translatey = 10
  var col = colorbrewer.RdPu
  var highlightcol = "darkred"
  var lineopacity = 1
  var cr = 1.75
  var copacity = 1
  var dotcolor = "black"
  var drawgrid = true

  function renderStackedGraph(svg) {

		var dwidth  = 800
		var dheight = 170

		var margin = {right: 23, left: 10, top: 10, bottom: 10}
		var width  = dwidth - margin.left - margin.right
		var height = dheight - margin.top - margin.bottom;

		var graphsvg = svg.append("g").attr("transform", "translate(" + translatex + "," + translatey + ")")

		var stack = zeros2D(n,m)
		var axisheight = 10
		var X = d3.scaleLinear().domain([0,stack.length]).range([margin.right,margin.right + width])
		var Y = d3.scaleLinear().domain([axis[1],axis[0]]).range([0,height])

		function add(a, b) { return a + b; }

		var s = []
		for (var j = 0; j < m; j ++) {

		  var si = graphsvg.append("g")

		  si.selectAll("line")
		    .data(stack)
		    .enter()
		    .append("line")
		    .attr("x1", function (d,i) { return X(i) } )
		    .attr("x2", function (d,i) { return X(i) } )
		    .attr("y1", function (d,i) { return Y(0) } )
		    .attr("y2", function (d,i) { return Y(d[0]) } )
		    .attr("stroke-width",2)
		    .attr("stroke", col[3][j])
        .attr("opacity", lineopacity)

		  s.push(si)

		}

		graphsvg.append("g").selectAll("circle")
		.data(stack)
		.enter()
		.append("circle")
		.attr("cx", function (d,i) { return X(i) } )
		.attr("cy", function (d,i) { return Y(d.reduce(add,0)) } )
		.attr("r", 2)
    .attr("opacity", copacity)

		function updateGraph(stacknew, highlight) {

			var svgdata = graphsvg.selectAll("circle").data(stacknew)
			svgdata.enter().append("circle")
			svgdata.merge(svgdata)
			  .attr("cx", function (d,i) { return X(i) } )
			  .attr("cy", function (d,i) { return Y(d.reduce(add,0)) } )
			  .attr("r",  function (d,i) { return highlight.includes(i) ? 2 : cr })
			  .attr("fill", function (d,i) { return highlight.includes(i) ? highlightcol : dotcolor })
			svgdata.exit().remove()

			for (var j = 0; j < m; j++) {
		    var svgdatai = s[j].selectAll("line").data(stacknew)
		    svgdatai.enter().append("line")
		    svgdatai.merge(svgdatai)
			    .attr("y1", function (d,i) { return Y(d.slice(0,j).reduce(add, 0)) } )
			    .attr("y2", function (d,i) { return Y(d.slice(0,j+1).reduce(add, 0)) } )
		    svgdatai.exit().remove()
			}

		}

    if (drawgrid) {
  		graphsvg.append("g")
  			.attr("class", "grid")
  			.attr("transform", "translate(0," + (height+10) + ")")
  			.attr("opacity", 0.25)
  			.call(d3.axisBottom(X)
  			  .ticks(5)
  			  .tickSize(2))

  		graphsvg.append("g")
  			.attr("class", "grid")
  			.attr("transform", "translate(12,0)")
  			.attr("opacity", 0.25)
  			.call(d3.axisLeft(Y)
  			  .ticks(0)
  			  .tickSize(2))
    }
		return {update: updateGraph, stack: s, X:X}

	}

  renderStackedGraph.highlightcol = function(_) {
  	highlightcol = _;
  	return renderStackedGraph;
  }

  renderStackedGraph.translatex = function(_) {
  	translatex = _;
  	return renderStackedGraph;
  }

  renderStackedGraph.translatey = function(_) {
  	translatey = _;
  	return renderStackedGraph;
  }

  renderStackedGraph.col = function(_) {
  	col = _
  	return renderStackedGraph
  }

  renderStackedGraph.lineopacity = function(_) {
    lineopacity = _
    return renderStackedGraph
  }

  renderStackedGraph.cr = function(_) {
    cr = _
    return renderStackedGraph
  }

  renderStackedGraph.copacity = function(_) {
    copacity = _
    return renderStackedGraph
  }

  renderStackedGraph.dotcolor = function(_) {
    dotcolor = _
    return renderStackedGraph
  }

  renderStackedGraph.drawgrid = function(_) {
    drawgrid = _
    return renderStackedGraph
  }
  return renderStackedGraph
}

/* 2d "scatterplot" with lines generator */
function plot2dGen(X, Y, iterColor) {

  var cradius = 1.2
  var copacity = 1
  var pathopacity = 1
  var pathwidth = 1
  var strokecolor = "black"

  function plot2d(svg) {

      var svgpath = svg.append("path")
        .attr("opacity", pathopacity)
        .style("fill", "none")
        .style("stroke", strokecolor)
        .style("stroke-width",pathwidth)
        .style("stroke-linecap","round")

      var valueline = d3.line()
        .x(function(d) { return X(d[0]); })
        .y(function(d) { return Y(d[1]); });

      var svgcircle = svg.append("g")

      var update = function(W) {

        // Update Circles
        var svgdata = svgcircle.selectAll("circle").data(W)

        svgdata.enter().append("circle")
          .attr("cx", function (d) { return X(d[0]) })
          .attr("cy", function (d) { return Y(d[1]) })
          .attr("r", cradius )
          .style("box-shadow","0px 3px 10px rgba(0, 0, 0, 0.4)")
          .attr("opacity", copacity)
          .attr("fill", function(d,i) { return iterColor(i)} )

        svgdata.merge(svgdata)
          .attr("cx", function (d) { return X(d[0]) })
          .attr("cy", function (d) { return Y(d[1]) })
          .attr("r", cradius )
          .attr("opacity", copacity)
          .attr("fill", function(d,i) { return iterColor(i)})
        svgdata.exit().remove()

        // Update Path
        svgpath.attr("d", valueline(W))

      }

      return update
  }

  // var cradius = 1.2
  // var copacity = 1
  // var pathopacity = 1
  // var pathwidth = 1

  plot2d.circleRadius = function(_) {
    cradius = _; return plot2d
  }

  plot2d.stroke = function(_) {
    strokecolor = _; return plot2d
  }

  plot2d.circleOpacity = function(_) {
    copacity = _; return plot2d
  }

  plot2d.pathWidth = function(_) {
    pathwidth = _; return plot2d
  }

  plot2d.pathOpacity = function(_) {
    pathopacity = _; return plot2d
  }

  plot2d.stroke = function (_) {
    strokecolor = _; return plot2d;
  }

  return plot2d
}

/* Render heatmap of f with colormap cmap onto canvas*/
function renderHeatmap(canvas, f, cmap) {
  var canvasWidth  = canvas.width;
  var canvasHeight = canvas.height;
  var ctx = canvas.getContext('2d');
  var imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);

  var buf = new ArrayBuffer(imageData.data.length);
  var buf8 = new Uint8ClampedArray(buf);
  var data = new Uint32Array(buf);

  for (var i = 0; i < canvasHeight; ++i) {
      for (var j = 0; j < canvasWidth; ++j) {
          var value = parseColor(cmap(f(j/canvasWidth, i/canvasHeight)))
          data[i * canvasWidth + j] =
              (255 << 24) |        // alpha
              (~~value[2] << 16) | // blue
              (~~value[1] <<  8) | // green
              ~~value[0];          // red
      }
  }
  imageData.data.set(buf8);
  ctx.putImageData(imageData, 0, 0);
}


function slider2D(div, onChange, lambda1, lambdan, start) {

	var panel = div.append("svg")
	              .append("g")
	              .attr("transform", "translate(25,30)")

	var width = 105
	var maxX = 4
	var maxY = 1

	var X = d3.scaleLinear()
	          .domain([0, maxX])
	          .range([0, 2*width])
	          .clamp(true);

	var Y = d3.scaleLinear()
	          .domain([maxY,0])
	          .range([0, width])
	          .clamp(true);

	var path = panel.append("path")
	            .attr("d","M 0 0 L " + 2*width + " 0 L " + width + " " + width + " L 0 " + width + " z")
	            .attr("fill","#EEE")
	            .attr("stroke", "#EEE")
	            .attr("stroke-width", 5)
	            .attr("stroke-linejoin", "round")
	            .on("click", function() {
	              var pt = d3.mouse(path.node())
	              var xy = clip(pt)
	              changeMouse(xy[0],xy[1])
	            })
	            .call( d3.drag().on("drag", function() {
	              var pt = d3.mouse(path.node())
	              var xy = clip(pt)
	              changeMouse(xy[0],xy[1])
	            }))

	var ly = panel.append("line")
	          .attr("x1", 0)
	          .attr("y1",0)
	          .attr("x2",width*2)
	          .attr("y2",0)
	          .style("stroke", "#DDD")
	          .style("stroke-width","3px")

	var lx = panel.append("line")
	          .attr("x1", 0)
	          .attr("y1",0)
	          .attr("x2",0)
	          .attr("y2",width)
	          .style("stroke", "#DDD")
	          .style("stroke-width","3px")

	var xval = 10
	var yval = 10

	function clip(pt) {
	  var y = Math.min(Math.max(0,pt[1]),width)
	  var x = Math.min(Math.max(0,pt[0]),2*width - y)
	  return [x,y]
	}

	function changeMouse(x,y) {
	  circle.attr("cx",x)
	  circle.attr("cy",y)
	  ly.attr("y1", y).attr("y2",y).attr("x1", 0).attr("x2",2*width-y)
	  lx.attr("x1", x).attr("x2",x).attr("y1", 0).attr("y2",(x>width) ? (width - (x-width)) : width)
	  onChange(X.invert(x),Y.invert(y))
	}

	var circle = panel.append("circle").attr("r", 7).attr("fill", "rgb(255, 102, 0)").style("stroke", "white").call(
	    d3.drag().on("drag", function() {
	      var pt = d3.mouse(path.node())
	      var xy = clip(pt)
	      changeMouse(xy[0],xy[1])
	    }))

	var tickwidth = 2
	var tickheight = 6
	var tickgroupX = panel.append("g").attr("transform", "translate(0, " + (-12) +")")

	tickgroupX.selectAll("rect")
	  .data([0,maxX/2,maxX], function(d,i) {return i})
	  .enter().append("rect")
	  .attr("x", function(i) { return isNaN(i) ? -100: X(i) - tickwidth/2})
	  .attr("y", 0)
	  .attr("width", tickwidth )
	  .attr("height", tickheight )
	  .attr("opacity",0.2 )

	tickgroupX.selectAll("text")
	  .data([0,maxX/2,maxX], function(d,i) {return i})
	  .enter().append("text")
	    .attr("class", "ticktext")
	    .attr("opacity", 0.3)
	    .attr("text-anchor", "middle")
	      .attr("transform", function(i) { return "translate(" + (X(i) - tickwidth/2 + 1) + "," + (tickwidth*2 -8) + ")" })
	      .html(function(d,i) { return d})

	var tickgroupY = panel.append("g").attr("transform", "translate(" + -12 + ", 0)")

	tickgroupY.selectAll("rect")
	  .data([0,maxY], function(d,i) {return i})
	  .enter().append("rect")
	  .attr("x", 0)
	  .attr("y", function(i) { return isNaN(i) ? -100: Y(i) - tickwidth/2})
	  .attr("width", tickheight )
	  .attr("height", tickwidth )
	  .attr("opacity",0.2 )

	tickgroupY.selectAll("text")
	  .data([0,maxY], function(d,i) {return i})
	  .enter().append("text")
	    .attr("class", "ticktext")
	    .attr("opacity", 0.3)
	    .attr("text-anchor", "middle")
	      .attr("transform", function(i) { return "translate(" + (-8) + "," + (Y(i) - tickwidth/2 + 5) + ")" })
	      .html(function(d,i) { return d})

	var beta = (Math.sqrt(lambda1) - Math.sqrt(lambdan))/(Math.sqrt(lambda1) + Math.sqrt(lambdan)); beta = beta*beta
	var alpha = 2/(Math.sqrt(lambda1) + Math.sqrt(lambdan))
	alpha = alpha*alpha

	var specialpoints = [[alpha*lambdan, beta]]

	panel.append("g").selectAll("circle")
	  .data(specialpoints)
	  .enter().append("circle")
	  .attr("cx", function(d,i) { return X(d[0])})
	  .attr("cy", function(d,i) { return Y(d[1])})
	  .attr("r", 2 )
	  .attr("opacity",0.2 )
	  .style("cursor", "pointer")
	  .on("click", function(d) { changeMouse(X(d[0]), Y(d[1]) ) })

	if  (!(start === undefined)) {
		changeMouse(X(start[0]),Y(start[1]))
	} else {
	  changeMouse(X(alpha*lambdan),Y(beta))
	}

  circle.moveToFront()

  return function() { changeMouse(X(specialpoints[0][0]), Y(specialpoints[0][1])) }
}

/****************************************************************************
  OPTIMIZATION RELATED FUNCTIONS
****************************************************************************/

/*
  Solves the system (1 - Lambda[i]*alpha)^k <= 1e-7
  for k.
*/
function getStepsConvergence(Lambda, alpha) {
  return Lambda.map( function(lambdai) {
    var o = -3*(1/Math.log10(Math.abs(1- lambdai*alpha)))
    return o < 0 ? NaN : o
  })
}

/*
  Run Momentum the good old fashioned way - by iterating.
  > runMomentum(bananaf, [0,0], 0.00001, 0.5, 100)
*/
function runMomentum(f, w0, alpha, beta, totalIters) {
  var Obj = []; var W = []; var z = zeros(w0.length); var w = w0
  var fx = f(w0); var gx = fx[1]
  W.push(w0); Obj.push(fx[0])
  for (var i = 0; i < totalIters; i++) {
    var z = numeric.add(numeric.mul(beta, z), gx)
    var w = numeric.add(w, numeric.mul(-alpha, z))
    fx = f(w); gx = fx[1]
    if (w.every(isFinite)) {
      W.push(w); Obj.push(fx[0])
    } else{ break; }
  }
  return [Obj, W]
}

/*
Closed form solution to the gradient descent iteration

w+ = w - alpha*([U*Lambda*U']*w - b)

Usage

iter = getiter(U,Lambda, b, alpha)
w    = iter(1000) <- gets the 1000'th iteration.
*/
function geniter(U, Lambda, b, alpha) {

  var Ub = numeric.dot(U,b)
  return function(k) {
    var c = []
    for (var i = 0; i < U.length; i++){
      var ci = Ub[i]*(1 - Math.pow(1 - alpha*Lambda[i],k))/Lambda[i]
      c.push(ci)
    }
    return numeric.dot(c, U)
  }
}

/*
  Generates function which computes the matrix geometric series
  sum([A^i for i = 0:(k-1)])*b
*/
function matSum(R,b) {

  // Buxfix hack for buggy numeric.js code.
  function fix(U) {
    if (U["y"] === undefined) {
      if (typeof U["x"][0] == "number") {
        U["y"] = zeros(U["x"].length)
      } else {
        U["y"] = zeros2D(U["x"].length, U["x"][0].length)
      }
    }
  }

  // Complex Diag functionality
  function diag(A) {
    var X = numeric.diag(A["x"])
    var Y = numeric.diag(A["y"])
    return new numeric.T(X,Y)
  }

  // Hack for tacking powers of complex numbers
  function pow(x,k) {
    return x.log().mul(k).exp()
  }

  var eR     = numeric.eig(R)
  // U*lambda*inv(U) = R
  var lambda = eR["lambda"]; fix(lambda)
  var U      = eR["E"]; fix(U)
  var bc     = new numeric.T(numeric.transpose([b]), zeros2D(b.length,1))
  var Uinvb  = U.inv().dot(bc); fix(Uinvb)
  // console.log(numeric.prettyPrint(R))
  // console.log(numeric.prettyPrint(lambda))
  // console.log(1-numeric.norm2([lambda.y[0],lambda.x[0]]), 1-numeric.norm2([lambda.y[1],lambda.x[1]]))
  return {matSum: function(k) {
    var topv = pow(lambda,k).mul(-1).add(1)
    var botv = lambda.mul(-1).add(1)
    var sumk = topv.mul(pow(botv,-1))
    if (lambda.getRow(0)["x"] == 1) { sumk["x"][0] = k; sumk["y"][0] = 0}
    if (lambda.getRow(1)["x"] == 1) { sumk["x"][1] = k; sumk["x"][1] = 0}
    return numeric.transpose(U.dot(diag(sumk)).dot(Uinvb)["x"])[0]
  }, lambda:(numeric.norm2([lambda.y[0],lambda.x[0]]))}

}

/*
Matrix power
*/
// function matPow(A) {

//   function fix(U) {
//     if (U["y"] === undefined) {
//       if (typeof U["x"][0] == "number") {
//         U["y"] = zeros(U["x"].length)
//       } else {
//         U["y"] = zeros2D(U["x"].length, U["x"][0].length)
//       }
//     }
//   }

//   // Complex Diag functionality
//   function diag(A) {
//     var X = numeric.diag(A["x"])
//     var Y = numeric.diag(A["y"])
//     return new numeric.T(X,Y)
//   }

//   // Hack for tacking powers of complex numbers
//   function pow(x,k) {
//     return x.log().mul(k).exp()
//   }

//   var eR     = numeric.eig(A)
//   // var lambda = eR["lambda"]; fix(lambda)
//   // var U      = eR["E"]; fix(U)
//   // var Uinv   = U.inv(); fix(Uinv)

//   return {matPow: function(k) {
// //    var lambdak = pow(lambda,k)
// //    return U.dot(diag(lambdak)).dot(Uinv)["x"]
//   }, lambda:(numeric.norm2([lambda.y[0],lambda.x[0]]))}

// }


/*
  Closed form solution for momentum iteration

  z_0 = zeros
  w_0 = zeros

  z+ = beta*z + ([U*Lambda*U']*w - b)
  w+ = w - alpha*z+

  Usage

  iter = getiter(U,Lambda, b, alpha)
  w    = iter(1000) <- gets the 1000'th iteration.
*/
function geniterMomentum(U, Lambda, b, alpha, beta) {

  var Ub = numeric.mul(-1,numeric.dot(U,b))

  var Rmat = function f(i) {
    return [[ beta          , Lambda[i]          ],
            [ -1*alpha*beta , 1 - alpha*Lambda[i]]]
  }

  var S = numeric.inv( [[1, 0], [alpha, 1]])

  var fcoll = []
  var maxLambda = []
  for (var i = 0; i < b.length; i++) {
  	m = matSum(Rmat(i),numeric.dot(S,[Ub[i],0]))
    fcoll.push(m.matSum)
    maxLambda.push(m.lambda)
  }

  return {iter: function(k) {
    var o = []
    for (var i = 0; i < b.length; i++) {
      o.push(fcoll[i](k))
    }
    return numeric.dot(numeric.transpose(o),U)
  }, maxLambda:maxLambda}
}

/* Returns the path and coordinates of annotation for a circle-annotation */
function ringPathGen(radius, width, height) {

  var padding = 4

  function ringPath(p1, p2) {

    // Generate Paths
    var x = -(p1[0] - p2[0])
        y = -(p1[1] - p2[1])
        xSign = (x > 0) - (x < 0),
        ySign = (y > 0) - (y < 0),
        r = radius,
        d = "",
        a = Math.sqrt(r * r / 2),
        b = Math.sqrt(r * r - Math.min(y * y, x * x)),
        c = Math.sqrt(2 * Math.min(x * x, y * y));
        dir = ""
    if (x * x + y * y < r * r) {
      d = "";
    } else if (c < r) {
      if (Math.abs(x) > Math.abs(y)) {
        dir = xSign > 0 ? "E" : "W"
        d = "M" + (p1[0] + xSign * b) + "," + (p1[1] + y)
          + ",L" + (p1[0] + x) + "," + (p1[1] + y)
      } else {
        dir = ySign > 0 ? "S" : "N"
        d = "M" + (p1[0] + x) + "," + (p1[1] + ySign * b)
         + ",L" + (p1[0] + x) + "," + (p1[1] + y)
      }
    } else if (Math.abs(x) > Math.abs(y)){
      dir = xSign > 0 ? "E" : "W"
      d = "M"  + (p1[0] + xSign * a)           + "," + (p1[1] + ySign * a) +
          ",L" + (p1[0] + xSign * Math.abs(y)) + "," + (p1[1] + y) +
          "L"  + (p1[0] + x)                   + "," + (p1[1] + y);
    } else {
      dir = ySign > 0 ? "S" : "N"
      d = "M"  + (p1[0] + xSign * a) + "," + (p1[1] + ySign * a) +
          ",L" + (p1[0] + x)         + "," + (p1[1] + ySign * Math.abs(x)) +
          "L"  + (p1[0] + x)         + "," + (p1[1] + y);
    }

    var top = 0
    var left = 0

    // Generate Paths
    // if (dir == "S") { top  = (p2[1] + padding); left = (p2[0] - width/2) }
    // if (dir == "N") { top  = (p2[1] - height - padding); left = (p2[0] - width/2) }
    // if (dir == "W") { top  = (p2[1] - height/2); left = (p2[0] - width - padding) }
    // if (dir == "E") { top  = (p2[1] - height/2); left = (p2[0] + padding) }

    if (dir == "S") { top  = (p2[1] + height/2 + padding); left = (p2[0] - width/2) }
    if (dir == "N") { top  = (p2[1] - padding); left = (p2[0] - width/2) }
    if (dir == "W") { top  = (p2[1] + height/4); left = (p2[0] - width - padding) }
    if (dir == "E") { top  = (p2[1] + height/4); left = (p2[0] + padding) }

    return {d:d, label:[left, top]}

  }

  return ringPath

}

function colorMap(root, width, colorScale, axisScale) {

  var margin = { top: 0, right: 12, bottom: 30, left: 12 };
  var height = 12;

  root.style("width", (width + margin.right + margin.left)  + "px")
  root.style("height", (height + margin.top + margin.bottom) + "px")
  var canvas = root.append("canvas")
      .attr("width", width+1)
      .attr("height", height)
      .style("position", "relative")
      .style("left", margin.left + "px")
      .style("top", "8px")
  var svg = root.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", 40)
      .style("left", -margin.left + "px")
      .style("opacity", 0.5)
    .append("g")
      .attr("transform", "translate(" + margin.left + ", 0)")
      .attr("class", "figtext")
  var axis = d3.axisBottom(axisScale).ticks(5);
  svg.call(axis);
  root.select(".label").text("Activation value")
  var context = canvas.node().getContext("2d");
  for (var i = 0; i < width + 1; i++) {
    context.fillStyle = colorScale(axisScale.invert(i));
    context.fillRect(i, 0, 1, height)
  }

}

function sliderBarGen(barlengths, expfn) {

	var update = function() {}
	var height = 60
	var maxX = 14.2
	var modval = 6
	var strokewidth = 2
	var gap = 15
	var mouseover = function(i) {}
	var labelFunc = function (d,i) { return ((i == 0) ? "Eigenvalue 1" : "") + (( (i+1) % modval == 0 ) ? (i + 1) : "")  }

	function sliderBar(div) {

	  var slider = div.append("div")
	                 .style("position", "relative")

	  var updateEverything = function(i, circ) {
	      step.html("Step k = " + numberWithCommas(Math.floor(expfn(i))) )

	      if (!(circ === undefined) ){
	        var ctm = circ.node().getCTM()
	        setTM(line.node(), ctm)
	        var barnodes = bars.nodes()
	        for (var j = 0; j < barlengths.length; j++) {
	          var r = d3.scaleLinear().domain([0,barlengths[j]-0.01,barlengths[j]-0.01,barlengths[j], 1/0]).range([1,1,1,0.2, 0.2])
	          d3.select(barnodes[j]).attr("opacity",r(i))
	          if (i > barlengths[j]) {
	             d3.select(barnodes[j]).style("stroke","black")
	          } else{
	             d3.select(barnodes[j]).style("stroke","black")
	          }
	        }
	      }

	      update(i)
	    }

	  var slidera = sliderGen([940, 60])
	    .ticks([0, maxX/5, 2*maxX/5, 3*maxX/5, 4*maxX/5, maxX])
	    .ticktitles(function(d) {return numberWithCommas(Math.floor(expfn(d))) })
	    .cRadius(7)
	    .startxval(4)
	    .shifty(3)
	    .margin({right: 160, left: 140})
	    .change(updateEverything)
	    (slider)

	  var width  = 695+80
	  var svg = slider.select("svg")

	  var step = svg.append("text").attr("class","figtext").attr("x",145).attr("y",15).html("Step k = ")

	  var x = d3.scaleLinear().domain([0,maxX, 100000]).range([90, width-45,width+45]);
	  var y = d3.scaleLinear().domain([0,barlengths.length]).range([10, height]);

	//  line.moveToBack()

	  var line = svg.append("line")
	     .attr("x1", 0 )
	     .attr("y1", 0)
	     .attr("x2", 0)
	     .attr("y2", height+50+gap)
	     .style("stroke", "black")
	     .style("stroke-width", "1px")

	  line.moveToBack()
	  svg.moveToFront()

	  var chart = svg.style("width", 940 + "px")
	                 .style("height", (height+100) + "px")
	                 .style("top", "30px")
	                 .append("g")
	                 .attr("transform", "translate(50, " + (gap+60) +" )")


	  chart.selectAll("rect").data(barlengths)
	     .enter()
	     .append("rect")
	     .attr("x", x(0) )
	     .attr("y", function(d,i) {return y(i)-2})
	     .attr("width", x(maxX) - 90)
	     .attr("height", 4)
	     .attr("opacity", 0.01)
	     .style("fill", "gray")
	     .on("mouseover", mouseover)
	     .on("mouseout", function(i) { update(slidera.xval()) })

	  var bars = chart.selectAll("line").data(barlengths)
	     .enter()
	     .append("line")
	     .attr("x1", x(0) )
	     .attr("y1", function(d,i) {return y(i)})
	     .attr("x2", function(d,i) {return x(d)})
	     .attr("y2", function(d,i) {return y(i)})
	     .style("stroke-width", strokewidth + "px")

	  chart.selectAll("text").data(barlengths)
	       .enter()
	       .append("text")
	       .attr("class", "figtext2")
	       .attr("text-anchor", "end")
	       .attr("x", 75)
	       .attr("y", function(d,i) {return y(i) + 4})
	       .style("width", 150 + "px")
	       .attr("fill", "gray")
	       .html(labelFunc)

	  chart.moveToBack()

	  function updateBars(barlengths_in) {
	  	for (var i = 0; i < barlengths.length; i++) { barlengths[i] = barlengths_in[i] }
	  	chart.selectAll("line").data(barlengths_in).merge(chart).attr("x2", function(d,i) {return x(d)})
	  }

	  return {slidera:slidera, update:updateBars}
	}

	sliderBar.height = function(_) {
		height = _; return sliderBar;
	}

	sliderBar.linewidth = function(_) {
		strokewidth = _; return sliderBar;
	}

	sliderBar.maxX = function(_) {
		maxX = _; return sliderBar;
	}

	sliderBar.update = function(_) {
		update = _; return sliderBar;
	}

	sliderBar.mouseover = function(_) {
		mouseover = _; return sliderBar;
	}

	sliderBar.mouseout = function(_) {
		mouseout = _; return sliderBar;
	}

	sliderBar.labelFunc = function(_) {
		labelFunc = _; return sliderBar;
	}

	return sliderBar
}

function renderDraggable(svg, p1, p2, radius, text) {

  var group = svg.append("g")
  var path = group.append("path").attr("fill", "none").attr("stroke","black").attr("stroke-width", 1);
  var circlePointer = svg.append("circle")
              .attr("cx", p1[0])
              .attr("cy", p1[1])
              .attr("r", radius)
              .attr("fill", "white").attr("fill-opacity",0).attr("stroke","black").attr("stroke-width", 1)
              .call(d3.drag().on("drag", function() {
                  var x = d3.mouse(this)[0]
                  var y = d3.mouse(this)[1]
                  p1 = [x,y]
                  var d = ringPath(p1, p2)
                  path.attr("d", d.d)
                  circlePointer.attr("cx",x).attr("cy",y)
                  console.log(p1, p2)
              }));

  var circleDragger = svg.append("circle")
              .attr("cx", p2[0])
              .attr("cy", p2[1])
              .attr("r", 4)
              .attr("fill", "white").attr("fill-opacity",0).attr("stroke","black").attr("stroke-width", 0)
              .call(d3.drag().on("drag", function() {
                  var x = d3.mouse(this)[0]
                  var y = d3.mouse(this)[1]
                  p2 = [x,y]
                  var d = ringPath(p1, p2)
                  path.attr("d", d.d)
                  label.attr("transform", "translate(" + d.label[0] + "," + d.label[1]  + ")")
                  circleDragger.attr("cx",x).attr("cy",y)
                  console.log(p1, p2)
              }));

  var label = svg.append("text")
      .style("position", "absolute")
      .style("border-radius", "3px")
      .style("text-align", "start")
      .style("padding-top", "5px")
      .attr("class", "figtext")
      .attr("x", 0)
      .attr("y",0)
      .attr("width", 100)
      .attr("height", 10)
      .attr("r", 7)
      .html(text)

  var ringPath = ringPathGen(radius, label.node().getBBox().width, label.node().getBBox().height)

  var d = ringPath(p1, p2)
  path.attr("d", d.d)
  label.attr("transform", "translate(" + d.label[0] + "," + d.label[1]  + ")")
  circleDragger.attr("cx",p2[0]).attr("cy",p2[1])

  return group
}

/****************************************************************************
  MISC MATH AND JAVASCRIPT HELPERS
****************************************************************************/

/* Parse a color string from rgb(0,0,0) format */
function parseColor(input) {
  return input.split("(")[1].split(")")[0].split(",").map(function(i){return parseInt(i.trim())});
}

/* Rosenbrok Function banana function */
function bananaf(xy) {
  var s = 3
  var x = xy[0]; var y = xy[1]*s
  var fx   = (1-x)*(1-x) + 20*(y - x*x )*(y - x*x )
  var dfx  = [-2*(1-x) - 80*x*(-x*x + y), s*40*(-x*x + y)]
  var d2fx = [ [ 2 + 160*x*x - 80*(-x*x + y), -80*x], [ -80*x, 40  ] ]
  return [fx*1, dfx, d2fx]
}

/* Nonsmooth variation on Rosenbrok Banana Function */
function bananaabsf(xy) {
  var x = xy[0]; var y = xy[1]
  var fx   = (1-x)*(1-x) + 20*(y - Math.abs(x) )*(y - Math.abs(x) )
  var dfx  = [-2*(1-x) - 20*((x > 0) ? 1 : -1)*(y - Math.abs(x)), 20*(y - Math.abs(x)) ]
  return [fx, dfx]
}

/* Rotated Quadratic */
function quadf(xy) {
  var U = givens(Math.PI/4)
  var lambda = numeric.diag([1,100])
  var A = numeric.dot(numeric.transpose(U),numeric.dot(lambda, U))
  var dfx = numeric.dot(A,xy)
  var fx  = 0.5*numeric.dot(dfx,xy)
  return [fx, dfx]
}

/* Identity */
function eyef(xy) {
  var U = givens(0)
  var lambda = numeric.diag([1,100])
  var A = numeric.dot(numeric.transpose(U),numeric.dot(lambda, U))
  var dfx = numeric.dot(A,xy)
  var fx  = 0.5*numeric.dot(dfx,xy)
  return [fx, dfx]
}

/* Givens rotations */
var givens = function(theta) {
  var c = Math.cos(theta)
  var s = Math.sin(theta)
  return [[c, -s], [s, c]]
}

/* Global controller for float -> string conversion */
function round(x) {
  return x.toPrecision(3)
}

/* Moves a svg element to the front */
d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};

d3.selection.prototype.moveToBack = function() {
    return this.each(function() {
        var firstChild = this.parentNode.firstChild;
        if (firstChild) {
            this.parentNode.insertBefore(this, firstChild);
        }
    });
};

/*
  Generates array of zeros
*/
function ones(n) {
  return Array.apply(null, Array(n)).map(Number.prototype.valueOf,1);
}

/*
  Generates array of zeros
*/
function zeros(n) {
  return Array.apply(null, Array(n)).map(Number.prototype.valueOf,0);
}

/*
  Generates array of zeros
*/
function zeros2D(n,m) {
  var A = []
  for (var i = 0; i < n; i ++) {
    A.push(zeros(m))
  }
  return A
}


/*
Create Vandermonde matrix of size x and order degree
*/
function vandermonde(x, degree){
	A = zeros2D(x.length,degree + 1)
	for (var i = 0; i < x.length; i ++){
	  for (var j = 0; j < degree + 1; j ++) {
	    A[i][j] = Math.pow(x[i],j)
	  }
	}
	return A
}

/*
Evaluate a 1D polynomial
w[0]x[0] + ... + w[k]x[k], k = w.length
*/
function poly(w,x) {
	s = 0
	for (var i = 0; i < w.length; i++) { s = s + w[i]*Math.pow(x,i) }
	return s
}


/*
Evaluates the polynomial in range [-1.1, 1.1] at 1800 intervals
*/
function evalPoly(w) {
	var data = []
	for (var i = -900; i < 900; i++) {
	  data.push([i/800, 1*poly(w, i/800)])
	}
	return data
}


function setTM(element, m) {
	return element.transform.baseVal.initialize(element.ownerSVGElement.createSVGTransformFromMatrix(m))
}


function wrap(text, width) {
  text.each(function() {
    var text = d3.select(this),
        words = text.text().split(/\s+/).reverse(),
        word,
        line = [],
        lineNumber = 0,
        lineHeight = 1.1, // ems
        y = text.attr("y"),
        dy = parseFloat(text.attr("dy")),
        tspan = text.text(null).append("tspan").attr("x", 0).attr("y", y).attr("dy", dy + "em");
    while (word = words.pop()) {
      line.push(word);
      tspan.text(line.join(" "));
      if (tspan.node().getComputedTextLength() > width) {
        line.pop();
        tspan.text(line.join(" "));
        line = [word];
        tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
      }
    }
  });
}

var inv = function(lambda) { return 1/lambda }

function eigSym(X) {
  var Eig = numeric.eig(X)
  var lambda = Eig.lambda.x
  var U = numeric.transpose(Eig.E.x)
  var Z = d3.zip(U, lambda)
  Z.sort(function(a, b) { return b[1] - a[1]; });
  U = []
  lambda = []
  for (var i = 0; i < Z.length; i++) {
    U.push(Z[i][0])
    lambda.push(Z[i][1])
  }
  return {U:U, lambda:lambda}
}

// http://stackoverflow.com/questions/2901102/how-to-print-a-number-with-commas-as-thousands-separators-in-javascript
function numberWithCommas(x) {
    var parts = x.toString().split(".");
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return parts.join(".");
}

function drawAnnotations(figure, annotations) {

    var figwidth = figure.style("width")
    var figheight = figure.style("height")

    var svg = figure.append("svg")
                .style("width", figwidth)
                .style("height", figheight)
                .style("position", "absolute")
                .style("top","0px")
                .style("left","0px")
                .style("pointer-events","none")

    var swoopy = d3.swoopyDrag()
      .x(function(d){ return (d.x) })
      .y(function(d){ return (d.y) })
        .draggable(false)
        .annotations(annotations)

    var swoopySel = svg.append("g").attr("class", "annotatetext").call(swoopy)

    svg.append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', '-10 -10 20 20')
        .attr('markerWidth', 20)
        .attr('markerHeight', 20)
        .attr('orient', 'auto')
      .append('path')
        .attr('d', 'M-6.75,-6.75 L 0,0 L -6.75,6.75')
        .attr("transform", "scale(0.5)")

    swoopySel.selectAll('path').attr('marker-end', 'url(#arrow)')

    return swoopySel

}