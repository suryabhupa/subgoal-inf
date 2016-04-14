require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'xlua'
require 'image'

noutputs = 50

nfeats = 3 -- because image is RGB
width = 210
height = 320
ninputs = nfeats*width*height

nstates = {64, 64, 128}
dims = {49, 77}
filtsize = 5
poolsize = 2 

function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end	

function create_model()
  --  CNN
	local convnet = nn.Sequential()
	convnet:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.View(nstates[2]*dims[1]*dims[2]))

	convnet:add(nn.Dropout(0.5))
	convnet:add(nn.Linear(nstates[2]*dims[1]*dims[2], nstates[3]))
	convnet:add(nn.ReLU())
	convnet:add(nn.Linear(nstates[3], noutputs)) 

  convnet:apply(weights_init)

  -- One-hot encoding
	local onehot = nn.Sequential()
	onehot:add(nn.Linear(9, 10)) 

  -- ParallelTable 
	local parallel = nn.ParallelTable()
	parallel:add(onehot)
	parallel:add(convnet)

  -- Deconvolution
  nc = 3
  dstates = 100
  genfilters = 64

  local decnet = nn.Sequential()
  -- input to convolution
  decnet:add(nn.SpatialFullConvolution(1, 2, 8, 5, 8, 5))
  decnet:add(nn.ReLU(true))

  decnet:add(nn.SpatialFullConvolution(2, 3, 4, 7, 4, 7))
  decnet:add(nn.ReLU(true))

  decnet:apply(weights_init)

  --TODO: connect back into an image

	print('==> creating model')
	local model = nn.Sequential()
	model:add(parallel)
	model:add(nn.JoinTable(1)) -- 60x1
  model:add(nn.Reshape(1,6,10)) -- 1x6x10
  model:add(decnet) -- 3x210x320
  model:add(nn.Sigmoid())
  
	-- Add more layers here for deconvolution
	print(model)
	return model

end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function get_action_data(file)
	-- read in the acations in the file, line by line
	print("==> getting action data")
	data = {}

	-- open file, check for errors
	fh, err = io.open(file) 
	if err then print("Error in opening file: " .. file) return end 

	-- loop while there are lines in the file
	while true do
		-- read line, break if nil
		local line = fh:read()
		if line == nil then break end

		-- each line is a string like "1 0 0 0 0 0 0"
		-- need to split string by space, and then concat to table
		-- then convert table to tensor, and add to overall data table
		inp = {}
		for i in string.gmatch(line, "%d") do
			table.insert(inp, tonumber(i))
		end
		table.insert(data, torch.Tensor(inp))
	end
	return data
end

-- Loads frame filenames into a table.
function get_frame_data(dir)
	-- get all *.png files in the directory and sort them
	print("==> getting frame data")
	files = {}
	for file in paths.files(dir) do
		if file:find("png$") then
			-- note that this only appends the *filenames* to the table, not the actual images
			table.insert(files, paths.concat(dir, file))
		end
	end 

	table.sort(files, function (a,b) return a < b end) 
	return files
end

-- Loads action and frame data into tables
function make_training_data(action_data_file, frame_data_dir)
	print("==> preparing dataset")
	a_data = get_action_data(action_data_file)
	f_data = get_frame_data(frame_data_dir)

  --print(a_data)
	--print(f_data)
	return {a_data, f_data}
end

function test(model)
end


function main()
	print("==> starting main")
	model = create_model()
	-- training of autoencoder here
	training_data = make_training_data("ALE/doc/examples/record/game_actions.txt", "ALE/doc/examples/record/")
	a_data = training_data[1]
	f_data = training_data[2]

	model:training() -- put into training mode (dropout turns on)
	criterion = nn.BCECriterion()

	for epoch = 1,100 do
		-- one less than the last entry, because we compare to next element
		for n_iter = 1,10867 do
			input = {a_data[n_iter], image.load(f_data[n_iter])}
			output = image.load(f_data[(n_iter+1)])
      output = nn.Sigmoid():forward(output)
			print("processing image " .. n_iter)
      

		 	local loss = criterion:forward(model:forward(input), output)
      print(loss)

			model:zeroGradParameters()
			model:backward(input, criterion:backward(model.output, output))
			model:updateParameters(0.01) 
		end
	end

	torch.save("cps/model-" .. epoch .. ".dat", model)
end

main()

