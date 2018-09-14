/*
 * particle_filter.cpp
 *
 *  Created on: Sept 10, 2018
 *      Author: Tiffany Huang
 */

#include <random>
#include <map>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	/*  Tried different values ranging from 1 to 100 particles , the particles could localize the vehicle accurately until
	 num_particles = 5 and the x_err and y_err are ~ 0.21, theta_err ~ 0.007. When increased it to num_particles = 100, the x_err and y_err
	 decreased to ~ 0.11, theta_err ~ 0.004. Hence no need to select a very large number of particles (ex. 100) which might be computationally
	 expensive, instead we shall use a fair number of particles which meets the passing criteria and above the min particles
	 number doing the job accurately (5 particles). */
	num_particles = 15;
	particles.resize(num_particles);
	weights.resize(num_particles);

	// This line creates a normal (Gaussian) distribution for x, y, theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(size_t i =0; i < num_particles; i++)
	{
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1/num_particles;
		weights[i] = particles[i].weight;
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	const double disp_delta_vel = velocity * delta_t;
	const double disp_delta_yaw = yaw_rate * delta_t;
	const double vel_yaw = velocity / yaw_rate;

	for (int i = 0; i < num_particles; i++)
	{
		// Check if the change in theta will be negligible or considered
		if (fabs(yaw_rate) < 0.001)
		{
			particles[i].x += disp_delta_vel * cos(particles[i].theta);
	    particles[i].y += disp_delta_vel * sin(particles[i].theta);
    }
    else
		{
			const double theta_new 	= particles[i].theta + disp_delta_yaw;
	    particles[i].x 					+= vel_yaw * (sin(theta_new) 					- sin(particles[i].theta));
	    particles[i].y 					+= vel_yaw * (cos(particles[i].theta) - cos(theta_new));
	    particles[i].theta 			= theta_new;
		}
	  // Adding random Gaussian noise
	  particles[i].x 			+= dist_x(gen);
	  particles[i].y 			+= dist_y(gen);
	  particles[i].theta 	+= dist_theta(gen);
	}
}


void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

		for(size_t ob=0; ob < observations.size(); ob++)
		{
			int min_idx = -1;
			double min = 1000;
			double eclud_dist = 0;

			for(size_t lm=0; lm < predicted.size(); lm++)
			{

				eclud_dist = dist(predicted[lm].x_f, predicted[lm].y_f, observations[ob].x, observations[ob].y);

				if(eclud_dist < min){
					min = eclud_dist;
					min_idx = predicted[lm].id_i;
				}
			}
			observations[ob].id = min_idx;
		}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	LandmarkObs part_lm;

	for(size_t p=0; p < particles.size(); p++)
	{
		std::vector<LandmarkObs> map_observations;
		std::vector<Map::single_landmark_s> landmark_list_range;

		for(size_t ob =0; ob <  observations.size(); ob++)
		{
			LandmarkObs part_mapObs;
			part_mapObs.x = particles[p].x + (observations[ob].x*cos(particles[p].theta)) - (observations[ob].y*sin(particles[p].theta));
			part_mapObs.y = particles[p].y + (observations[ob].x*sin(particles[p].theta)) + (observations[ob].y*cos(particles[p].theta));

			map_observations.push_back(part_mapObs);
		}

	for(size_t l=0; l < map_landmarks.landmark_list.size(); l++)
	{
		if (dist(particles[p].x, particles[p].y, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f) < sensor_range)
		{
			float lm_x = map_landmarks.landmark_list[l].x_f;
			float lm_y = map_landmarks.landmark_list[l].y_f;
			int lm_id = map_landmarks.landmark_list[l].id_i;
			landmark_list_range.push_back(Map::single_landmark_s{lm_id, lm_x, lm_y});
		}
	}

		dataAssociation(landmark_list_range, map_observations);

		particles[p].weight = 1;
		for(size_t obs=0; obs < map_observations.size(); obs++)
		{
			for (size_t i; i< landmark_list_range.size(); i++)
			{
				if(landmark_list_range[i].id_i == map_observations[obs].id)
				{
					float lmx = landmark_list_range[i].x_f;
					float lmy = landmark_list_range[i].y_f;
				}
			}

			particles[p].weight *= (1.0 / (2*M_PI*std_landmark[0]*std_landmark[1])) * \
			(exp(-((pow((map_observations[obs].x-map_landmarks.landmark_list[map_observations[obs].id-1].x_f),2) / (2*pow(std_landmark[0],2)))+ \
			((pow((map_observations[obs].y-map_landmarks.landmark_list[map_observations[obs].id-1].y_f),2)) / (2*pow(std_landmark[1],2))))));

		}

		weights[p] = particles[p].weight;
	}

}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<int> d(weights.begin(), weights.end());
	std::vector<Particle> weighted_sample(num_particles);

	for(int i = 0; i < num_particles; ++i){
		int j = d(gen);
		weighted_sample.at(i) = particles.at(j);
	}

	particles = weighted_sample;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
