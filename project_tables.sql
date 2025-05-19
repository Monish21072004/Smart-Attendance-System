-- Create a new database called "smart_attendance"
CREATE DATABASE IF NOT EXISTS smart_attendance;

-- Use the newly created database
USE smart_attendance;

-- Create the Users table to store login credentials and profile information
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL,  -- Store hashed passwords
  role ENUM('teacher', 'student') NOT NULL,
  name VARCHAR(100),
  email VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the Attendance table to record attendance entries for students
-- The attendance_timestamp column automatically stores the current date and time upon record insertion
CREATE TABLE attendance (
  id INT AUTO_INCREMENT PRIMARY KEY,
  student_id INT NOT NULL,
  attendance_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  image_path VARCHAR(255),  -- Optionally store the path to the captured image
  FOREIGN KEY (student_id) REFERENCES users(id)
);
INSERT INTO users (username, password, role, name, email)
VALUES ('Monish1', 'blablacar', 'teacher', 'Monish', 'monish@gmail.com');

INSERT INTO users (username, password, role, name, email) VALUES ('akshay1', 'blabla', 'student', 'Akshay Kumar', 'akshay@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('alexandra1', 'blabla', 'student', 'Alexandra Daddario', 'alexandra@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('alia1', 'blabla', 'student', 'Alia Bhatt', 'alia@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('amitabh1', 'blabla', 'student', 'Amitabh Bachchan', 'amitabh@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('andy1', 'blabla', 'student', 'Andy Samberg', 'andy@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('anushka1', 'blabla', 'student', 'Anushka Sharma', 'anushka@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('billie1', 'blabla', 'student', 'Billie Eilish', 'billie@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('brad1', 'blabla', 'student', 'Brad Pitt', 'brad@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('camila1', 'blabla', 'student', 'Camila Cabello', 'camila@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('charlize1', 'blabla', 'student', 'Charlize Theron', 'charlize@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('claire1', 'blabla', 'student', 'Claire Holt', 'claire@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('courtney1', 'blabla', 'student', 'Courtney Cox', 'courtney@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('dwayne1', 'blabla', 'student', 'Dwayne Johnson', 'dwayne@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('elizabeth1', 'blabla', 'student', 'Elizabeth Olsen', 'elizabeth@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('ellen1', 'blabla', 'student', 'Ellen Degeneres', 'ellen@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('henry1', 'blabla', 'student', 'Henry Cavill', 'henry@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('hrithik1', 'blabla', 'student', 'Hrithik Roshan', 'hrithik@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('hugh1', 'blabla', 'student', 'Hugh Jackman', 'hugh@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('jessica1', 'blabla', 'student', 'Jessica Alba', 'jessica@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('kashyap1', 'blabla', 'student', 'Kashyap', 'kashyap@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('lisa1', 'blabla', 'student', 'Lisa Kudrow', 'lisa@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('margot1', 'blabla', 'student', 'Margot Robbie', 'margot@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('marmik1', 'blabla', 'student', 'Marmik', 'marmik@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('natalie1', 'blabla', 'student', 'Natalie Portman', 'natalie@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('priyanka1', 'blabla', 'student', 'Priyanka Chopra', 'priyanka@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('robert1', 'blabla', 'student', 'Robert Downey Jr', 'robert@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('roger1', 'blabla', 'student', 'Roger Federer', 'roger@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('tom1', 'blabla', 'student', 'Tom Cruise', 'tom@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('vijay1', 'blabla', 'student', 'Vijay Deverakonda', 'vijay@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('virat1', 'blabla', 'student', 'Virat Kohli', 'virat@gmail.com');
INSERT INTO users (username, password, role, name, email) VALUES ('zac1', 'blabla', 'student', 'Zac Efron', 'zac@gmail.com');
select * from attendance