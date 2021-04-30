setwd("~/PycharmProjects/STAT8002/r/data")
library(ggplot2)
library(ggpubr)
library(MVN)

test_equal_mean=function(sample_set1,sample_set2,alpha=0.05){
  mean1 = colMeans(sample_set1)
  print("First mean")
  print(mean1)
  mean2 = colMeans(sample_set2)
  print("Second mean")
  print(mean2)
  n1 = nrow(sample_set1)
  n2 = nrow(sample_set2)
  s1 = cov(sample_set1)
  s2 = cov(sample_set2)
  s = (s1*(n1-1)+s2*(n2-1))/(n1+n2-2)
  mean_diff = mean1-mean2
  print("Mean difference")
  print(mean_diff)
  p = nrow(s1)
  t_square = n1*n2/(n1+n2)*mean_diff%*%solve(s)%*%mean_diff
  print(sprintf("Computed Hotelling's T square = %f",t_square))
  m = n1+n2-p-1
  cv = (n1+n2-2)*p/m*qf(1-alpha,p,m)
  print(sprintf("Critical value = %f",cv))
  f_val = t_square*m/((n1+n2-2)*p)
  print(f_val)
  p_val = 1-pf(f_val,p,m)
  print(p_val)
  return(p_val)
}

experiment1_samples = read.csv('Experiment1_samples.csv')
experiment2_samples = read.csv('Experiment2_samples.csv')
experiment3a_samples = read.csv('Experiment3a_samples.csv')
experiment3b_samples = read.csv('Experiment3b_samples.csv')
accuracies = rbind(experiment1_samples,experiment2_samples,experiment3a_samples,experiment3b_samples)

# T+N trading day
ggplot(accuracies,aes(x=factor(DELAY),y=ACCURACY,fill=GRAIN))+geom_boxplot()

# Headline vs Content
convert_to_matrix<-function(experiment,text,delay,tool="VADER"){
  df0 <- subset(accuracies,EXPERIMENT==experiment&DELAY==delay&TEXT==text&TOOL==tool,select=c('GRAIN','ACCURACY'))
  df1 = subset(df0,GRAIN=="CORN",select=c("ACCURACY"))
  colnames(df1)="CORN"
  df2 = subset(df0,GRAIN=="SOYBEAN",select=c("ACCURACY"))
  colnames(df2)="SOYBEAN"
  df3 = subset(df0,GRAIN=="WHEAT",select=c("ACCURACY"))
  colnames(df3)="WHEAT"
  df4 = subset(df0,GRAIN=="RICE",select=c("ACCURACY"))
  colnames(df4)="RICE"
  df5 = subset(df0,GRAIN=="OAT",select=c("ACCURACY"))
  colnames(df5)="OAT"
  return(cbind(df1,df2,df3,df4,df5))
}
test_equal_mean(convert_to_matrix(1,"Headline",1),convert_to_matrix(1,"Content",1))
test_equal_mean(convert_to_matrix(2,"Headline",1),convert_to_matrix(2,"Content",1))
test_equal_mean(convert_to_matrix("3a","Headline",1),convert_to_matrix("3a","Content",1))
test_equal_mean(convert_to_matrix("3a","Headline",1,"SENTIWORDNET"),convert_to_matrix("3a","Content",1,"SENTIWORDNET"))
test_equal_mean(convert_to_matrix("3b","Headline",1),convert_to_matrix("3b","Content",1))
test_equal_mean(convert_to_matrix("3b","Headline",1,"SENTIWORDNET"),convert_to_matrix("3b","Content",1,"SENTIWORDNET"))

# Doc2Vec
test_equal_mean(convert_to_matrix("3b","Headline",1),convert_to_matrix("3a","Headline",1))
test_equal_mean(convert_to_matrix("3b","Content",1),convert_to_matrix("3a","Content",1))
test_equal_mean(convert_to_matrix("3b","Headline",1,"SENTIWORDNET"),convert_to_matrix("3a","Headline",1,"SENTIWORDNET"))
test_equal_mean(convert_to_matrix("3b","Content",1,"SENTIWORDNET"),convert_to_matrix("3a","Content",1,"SENTIWORDNET"))

# VADER vs SENTIWORDNET
test_equal_mean(convert_to_matrix("3a","Headline",1),convert_to_matrix("3a","Headline",1,"SENTIWORDNET"))
test_equal_mean(convert_to_matrix("3a","Content",1),convert_to_matrix("3a","Content",1,"SENTIWORDNET"))
test_equal_mean(convert_to_matrix("3b","Headline",1),convert_to_matrix("3b","Headline",1,"SENTIWORDNET"))
test_equal_mean(convert_to_matrix("3b","Content",1),convert_to_matrix("3b","Content",1,"SENTIWORDNET"))

