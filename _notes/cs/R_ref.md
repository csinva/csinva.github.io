```r
x %%2 # modulus
x <- 3 # assignment
class(x) checks the class of x
rm(list=ls())
```
- *vectors*
	- numeric_vector <- c(1, 2, 3)
	- poker_vector <- c(140, -50, 20, -120, 240)
	- names(poker_vector) <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
		- assigns names to elements of poker vector
- *matrices*
	- matrix(1:9, byrow = TRUE, nrow = 3)
	- can name the rows / cols
	- 1-indexed
	- has slicing
	- dim(m) prints dimensions
- *factor* - data type for storing categorical variable
- *data frame* - when you want different types of data
	- columns are variables, rows are observations
- *lists* - ordered, can hold any data type
	- length(list)