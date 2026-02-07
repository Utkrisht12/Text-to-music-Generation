# Define AWS provider and region
provider "aws" {
  region = "us-east-2"
}

variable "training_data_bucket" {
  description = "Name of the S3 bucket containing the training data"
  default = "idlmusicgeneration"
}

variable "args_python" {
  type    = list(string)
  default = ["semantic","coarse", "fine"] #["semantic", "coarse", "fine"]
}

variable "instance_types" {
  type    = list(string)
  default = ["p3.2xlarge", "g4dn.2xlarge", "g4dn.2xlarge"] #"g4dn.2xlarge" 	p3.2xlarge #EXPENSIVE
}


#Aws security groups
resource "aws_security_group" "trainer_sec_group" {
  name        = "trainer_sec_group"
  description = "allow ssh traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    cidr_blocks     = ["0.0.0.0/0"]
  }
}

# Create an Amazon EBS volume for training data from the snapshot
resource "aws_ebs_volume" "training_data_volume" {
  count             = length(var.args_python)
  availability_zone = "us-east-2a"
  size              = 700 # Change the volume size as per your requirement
  type              = "gp3"
  snapshot_id       =  "snap-04818e1040e3faac3" #"snap-04818e1040e3faac3"#full-data #"snap-07550f117af03163a" #ebs_partial_data_test

  tags = {
    Name = "TrainingDataVolume-${count.index + 1}"
  }
}

resource "aws_iam_instance_profile" "admin_profile" {
  name = "admin_profile"
  role = "admin"
}

# Create an Amazon EC2 instance for training
resource "aws_instance" "training_instance" {
  count = length(var.args_python) # Launch 3 instances for training in parallel

  ami           = "ami-0996d1ddefe09ff57" # Deep learning AMI with PyTorch
  instance_type = var.instance_types[count.index] #"g5.2xlarge" # GPU instance type for faster training
  vpc_security_group_ids = [
    aws_security_group.trainer_sec_group.id
  ] 
  availability_zone = "us-east-2a"
  # Use a user data script to install dependencies and start training
  user_data = <<-EOF
              #!/bin/bash
              (
              #Wait for attachment to be ready
              echo "Sleeping 10 sec"
              sleep 30
              echo "Resuming.."
            
              # Attach Amazon EBS volume for training data
              sudo mkdir /mnt/data
              sudo mount /dev/nvme2n1 /mnt/data || sudo mount /dev/nvme1n1 /mnt/data  || sudo mount /dev/xvdf /mnt/data
              sudo chown -R ubuntu:ubuntu /mnt/data

              sudo apt update
              sudo apt-get install sox -y
              sudo apt-get install ffmpeg -y

              # Download training script from S3 bucket
              aws s3 cp --recursive s3://${var.training_data_bucket}/trainer/ /home/ubuntu/

              # Install PyTorch and other dependencies
              sudo pip install -r /home/ubuntu/requirements.txt
              
              export $(cat .env | xargs) &> /dev/null
              
              #Purge huge files
              find /mnt/data/data/ -size +10M -delete
              # Start training script
              mkdir -p /mnt/data/results
              sudo chown -R ubuntu:ubuntu /mnt/data/results
              python3 /home/ubuntu/app.py ${var.args_python[count.index]} --run_number 6
              )  &> /home/ubuntu/log.txt &
              disown
              EOF

  root_block_device {
    delete_on_termination = true
    iops = 3000
    volume_size = 70
    volume_type = "gp3"
  }
  
  key_name = "second"
  iam_instance_profile = aws_iam_instance_profile.admin_profile.name
  tags = {
    Name = "TrainingInstance${count.index + 1}"
  }
  depends_on = [ aws_security_group.trainer_sec_group, aws_ebs_volume.training_data_volume]
}

#TODO: attach device inmediately
resource "aws_volume_attachment" "trainer_vol" {
    count = length(var.args_python)
    device_name = "/dev/xvdf"
    volume_id = "${aws_ebs_volume.training_data_volume[count.index].id}"
    instance_id = "${aws_instance.training_instance[count.index].id}"
}