#!/usr/bin/env RScript
# concatenate csv's from "segmentation_file.list"

filelist = read.table('segmentation_data.list')
df = data.frame()
for (i in 1:nrow(filelist)) {
  f = read.table(filelist[i,1], sep=',', header=T)
  df = rbind(df,f)
  
}
cat("Breakdown of data:\n\n")
cat("\tTotal data rows:", nrow(df),"\n")

cat("\nNumber of rows per genotype:")
print(table(df$Genotype))

cat("\nNumber of rows per Genotype/RNAi combination:")
print(table(df$Genotype, df$RNAi))

cat("\nNumber of rows per Genotype/Rep combination:")
print(table(df$Genotype, df$Rep))

cat("\nNumber of rows per Genotype/Rep/RNAi combination:\n")
print(table(df$Genotype, df$Rep, df$RNAi))

outputfile="all_dist_segmentation_results.csv"
write.table(df, outputfile, row.names=F, sep=",", quote=F)

cat("\n\nWrote", outputfile, "\n")
