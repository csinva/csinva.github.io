// view themes at https://highlightjs.org/static/demo/
s = '' // '<div class="header"> interpretable ml \t tests </div>'

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
              return  '\n----\n' + line + s;
          else
              return line;
        })
        /*
        .map((line, index) => {
            if (/<f>/.test(line)){
                if(line[0] == '-'){ // for bullets don't fade in the bullet
                    line = '- <span class="fragment fade-in">' + line.substr(2) + '</span>';
                }
                else{
                    line = '<span class="fragment fade-in">' + line + '</span>';
                }
            }
            return line
        })
        */
        /*
        .map((line, index) => {
            if (/<_f/.test(line))
                line = line.replace('<_f', '<span class="fragment fade-in">');
            if (/_f>/.test(line))
                line = line.replace('_f>', '</span>');
            return line
        })
        */
        .join('\n')
    );
  });
};