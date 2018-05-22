/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include "map.h"

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    default_random_engine gen;

    // Create normal distributions for x,y, theta for particle generation
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Add particles with random gaussian noise around given state x,y,theta
    particles.reserve(num_particles);
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.; // all particles are equally probable to be the actual vehicle state
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    // use vehicle motion model to predict new state of all particles
    double yaw_change = yaw_rate * delta_t;
    double velocity_over_yaw_rate = velocity / yaw_rate;
    for (Particle &p : particles) {
        double x_offset;
        double y_offset;

        // avoid division by zero
        if (fabs(yaw_rate < 0.00001)) {
            double distance_change = velocity * delta_t;
            x_offset = distance_change * cos(p.theta);
            y_offset = distance_change * sin(p.theta);
        } else {
            x_offset = velocity_over_yaw_rate * (sin(p.theta + yaw_change) - sin(p.theta));
            y_offset = velocity_over_yaw_rate * (cos(p.theta) - cos(p.theta + yaw_change));
        }

        p.x += x_offset;
        p.y += y_offset;
        p.theta += yaw_change;

        // add gaussian noise for motion
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the
    // observed measurement to this particular landmark.
    for (LandmarkObs &lm_obs : observations) {
        double dist_closest = std::numeric_limits<double>::max();
        LandmarkObs *lm_pred_closest = nullptr;
        for (LandmarkObs &lm_pred : predicted) {
            auto dist = sqrt(pow(lm_pred.x - lm_obs.x, 2) + pow(lm_pred.y - lm_obs.y, 2));
            if (dist < dist_closest) {
                dist_closest = dist;
                lm_pred_closest = &lm_pred;
            }
        }

        if (lm_pred_closest != nullptr) {
            int id_closed =  (*lm_pred_closest).id;
            lm_obs.id = id_closed;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // Update the weights of each particle using a multi-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html


    // create map id to landmark to speed up later referencing to landmarks for each particle and observation
    std::map<int, LandmarkObs> lm_predicted_by_id;
    for (auto const &lm : map_landmarks.landmark_list) {
        lm_predicted_by_id[lm.id_i] = LandmarkObs{lm.id_i, lm.x_f, lm.y_f};
    }

    const double std_lm_x = std_landmark[0];
    const double std_lm_y = std_landmark[1];
    const double std_lm_x_sq2_mul2 = 2 * pow(std_lm_x, 2);
    const double std_lm_y_sq2_mul2 = 2 * pow(std_lm_y, 2);
    const double gaussian_lm_divisor = (1. / (2. * M_PI * std_lm_x * std_lm_y));

    // update each partice's weight
    for (Particle &p : particles) {
        const double px = p.x;
        const double py = p.y;
        const double ptheta = p.theta;
        const double cos_ptheta = cos(ptheta);
        const double sin_ptheta = sin(ptheta);

        // Prepare landmarks in sensor range of particle in global coordinates.
        // Helps to reduce the computational effor for finding associations between map and observation landmarks.
        vector<LandmarkObs> lm_predicted_particle;
        for (auto const &item : lm_predicted_by_id) {
            const auto lm = item.second;
            const double lm_x = item.second.x;
            const double lm_y = item.second.y;
            const double dist_to_lm = sqrt(pow(px - lm_x, 2) + pow(py - lm_y, 2));
            const bool in_sensor_range = dist_to_lm <= sensor_range;
            if (in_sensor_range) {
                lm_predicted_particle.push_back(lm);
            }
        }

        // Convert observed landmarks with local coords in sensor range to global coords.
        vector<LandmarkObs> lm_obs_particle;
        for (const LandmarkObs &lm_obs : observations) {
            const double lm_obs_x = lm_obs.x;
            const double lm_obs_y = lm_obs.y;
            auto x_global = lm_obs_x * cos_ptheta - lm_obs_y * sin_ptheta + px;
            auto y_global = lm_obs_y * cos_ptheta + lm_obs_x * sin_ptheta + py;
            lm_obs_particle.push_back(LandmarkObs{lm_obs.id, x_global, y_global});
        }

        // find landmarks on map (predicted) for observed landmarks
        dataAssociation(lm_predicted_particle, lm_obs_particle);

        // update particle weight by using multi variate gaussian distribution
        double pweight = 1.;
        for (const LandmarkObs &lm_obs : lm_obs_particle) {
            auto obs_x = lm_obs.x;
            auto obs_y = lm_obs.y;
            auto lm_predicted = lm_predicted_by_id[lm_obs.id];
            if (lm_predicted.id == 0) {
                // handle if no map landmark was found for observed landmark
                pweight = 0;
                break;
            }
            auto pred_x = lm_predicted.x;
            auto pred_y = lm_predicted.y;
            auto diff_x = pred_x - obs_x;
            auto diff_y = pred_y - obs_y;
            double observation_weight = gaussian_lm_divisor * exp(-(pow(diff_x, 2) / std_lm_x_sq2_mul2 +
                    (pow(diff_y, 2) / std_lm_y_sq2_mul2)));
            pweight *= observation_weight;
        }
        p.weight = pweight;
    }

}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    auto max_weight = (*std::max_element(particles.begin(), particles.end(),
                                         [](const Particle &a, const Particle &b) {
                                             return a.weight < b.weight;
                                         })).weight;

    default_random_engine gen;
    double beta = 0.;
    std::uniform_int_distribution<int> index_dist(0, num_particles);
    auto index = index_dist(gen);

    std::uniform_real_distribution<double> weight_dist(0., max_weight);
    std::vector<Particle> particles_resampled;

    for (int i = 0; i < num_particles; i++) {
        beta += weight_dist(gen) * 2.;
        double weight;
        while (beta > (weight = particles[index].weight)) {
            beta -= weight;
            index = (index + 1) % num_particles;
        }
        particles_resampled.push_back(particles[index]);
    }

    particles = particles_resampled;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
