library("sva")
setwd("D:\\Ziyi\\School\\PMO\\metanalysis\\Simulation\\combine\\data")

data1<-read.table("Dataset1_series_matrix.txt",sep="\t",header=T,row.names=1)
data2<-read.table("Dataset2_series_matrix.txt",sep="\t",header=T,row.names=1)
data3<-read.table("Dataset3_series_matrix.txt",sep="\t",header=T,row.names=1)
sample<-read.table("Datasety.txt",sep = "\t",header = T)

dat<-cbind(data1,data2,data3)
rownames(sample)<-sample[,1]

# Combat
data_combat<-ComBat(dat, batch=sample[,2],mod=NULL, par.prior=TRUE, prior.plots=TRUE)
write.table(data_combat,file="data_combat.txt",sep="\t")
