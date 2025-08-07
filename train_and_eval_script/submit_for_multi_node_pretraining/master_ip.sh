# com: hostnamectl --static set-hostname guomy / hostname guomy

# cat /ect/hosts | grep guomy* | awk '{print $1}' > master_ip

cat /etc/hosts | awk 'END{print $1}'  > master_ip

# cat /etc/hosts | awk 'END{print}'  >> master_ip
# cat master_ip