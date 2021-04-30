setwd("~/PycharmProjects/STAT8002/python/data")
my_analysis <- function(filename, delay=0){
  d = read.csv(filename);
  v = as.numeric(sub("%", "", d$Change..))/100;
  v = rev(v);
  if(delay==0){
    return(v)
  }
  v = diff(v, differences = delay);
  return (ts(v));
}
my_ts <- function(filename){
  d = read.csv(file = filename);
  v = d$Price;
  v = rev(v);
  return (ts(v));
}
minMaxScale<-function(arr){
  return((arr-min(arr))/(max(arr)-min(arr)))
}
corn = data.frame('T+0'=corn1_min_max[1:1070],'T+1'=corn1_min_max[2:1071],'T+2'=corn1_min_max[3:1072])
m = lm(T.2~T.0+T.1,data = corn)