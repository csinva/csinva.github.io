// view themes at https://highlightjs.org/static/demo/
module.exports = (markdown, options) => {
  return new Promise((resolve, reject) => {
    return resolve(
      markdown
        .split('\n')
        .map((line, index) => {
          if (/### /.test(line))
              return line;
          else if (/## /.test(line)) 
              return  '\n---\n' + line;
          else if (/# /.test(line)) 
              return  '\n----\n' + line;
          else
              return line;
        }).join('\n')
    );
  });
};