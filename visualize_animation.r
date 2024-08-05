#!/usr/bin/env Rscript

# Load necessary libraries
if (!require(sf)) install.packages("sf", repos = "http://cran.us.r-project.org")
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if (!require(gganimate)) install.packages("gganimate", repos = "http://cran.us.r-project.org")
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if (!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if (!require(gifski)) install.packages("gifski")
if (!require(av)) install.packages("av")

library(sf)
library(ggplot2)
library(gganimate)
library(dplyr)
library(tidyr)
library(gifski)
library(av)

# Define paths

reach_info_path <- "./rapid_data/NHDFlowline_San_Guad/reach_info.csv"
discharge_path <- "./model_saved/discharge_est.csv"
riv_bas_id_path <- "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"

# Read the reach information CSV file
reach_info <- read.csv(reach_info_path)

# Read the discharge estimation CSV file
discharge_data <- read.csv(discharge_path, header = FALSE)
kf_id <- read.csv(riv_bas_id_path, header = FALSE)

# Prepare discharge data
cutoff <- 5175
discharge_data <- discharge_data[1:cutoff, ]
kf_id <- kf_id[1:cutoff, ]
widths <- t(discharge_data)

# Normalize widths
max_width <- max(widths)
normalized_widths <- (widths / max_width)^0.33 * 50

# Reduce number of frames for testing
max_frames <- 10
normalized_widths <- normalized_widths[1:max_frames, ]

# Convert reach information to Cartesian coordinates
R <- 6371000 # Earth's radius in meters (mean radius)

lat_lon_to_cartesian <- function(lat, lon, lat_ref, lon_ref) {
  x <- R * (lon - lon_ref) * cos(pi * lat_ref / 180)
  y <- R * (lat - lat_ref)
  return(data.frame(x = x, y = y))
}

ref_lat <- reach_info$Start.Latitude[1]
ref_long <- reach_info$Start.Longitude[1]

first_vertex <- lat_lon_to_cartesian(reach_info$Start.Latitude, reach_info$Start.Longitude, ref_lat, ref_long)
last_vertex <- lat_lon_to_cartesian(reach_info$End.Latitude, reach_info$End.Longitude, ref_lat, ref_long)

# Prepare data for plotting
plot_data <- data.frame()

for (frame in 1:nrow(normalized_widths)) {
    cat("Processing frame:", frame, "of", nrow(normalized_widths), "\n")
  frame_data <- data.frame(
    x1 = first_vertex$x + (- (last_vertex$y - first_vertex$y) * normalized_widths[frame, ] / 2),
    y1 = first_vertex$y + ((last_vertex$x - first_vertex$x) * normalized_widths[frame, ] / 2),
    x2 = last_vertex$x + (- (last_vertex$y - first_vertex$y) * normalized_widths[frame, ] / 2),
    y2 = last_vertex$y + ((last_vertex$x - first_vertex$x) * normalized_widths[frame, ] / 2),
    x3 = last_vertex$x - (- (last_vertex$y - first_vertex$y) * normalized_widths[frame, ] / 2),
    y3 = last_vertex$y - ((last_vertex$x - first_vertex$x) * normalized_widths[frame, ] / 2),
    x4 = first_vertex$x - (- (last_vertex$y - first_vertex$y) * normalized_widths[frame, ] / 2),
    y4 = first_vertex$y - ((last_vertex$x - first_vertex$x) * normalized_widths[frame, ] / 2),
    frame = frame
  )
  plot_data <- rbind(plot_data, frame_data)
}

# Plotting
plot_data <- plot_data %>%
  pivot_longer(cols = c(x1, y1, x2, y2, x3, y3, x4, y4), 
               names_to = c(".value", "coord"), 
               names_pattern = "(.)[1-4]")

gg <- ggplot() +
  geom_polygon(data = plot_data, aes(x = x, y = y, group = interaction(frame, coord)), fill = "blue", alpha = 0.6) +
  labs(x = "X (meters)", y = "Y (meters)", title = "River Width over Time") +
  theme_minimal() +
  transition_time(frame) +
  ease_aes('linear')

animate(gg, nframes = max_frames, fps = 10, renderer = gifski_renderer("river_animation.gif"))